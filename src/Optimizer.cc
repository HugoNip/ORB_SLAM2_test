/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include "Optimizer.h"

#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

#include<Eigen/StdVector>

#include "Converter.h"

#include<mutex>

/**
 * Optimizer是用来优化的类，ORB SLAM2中所有优化的方法都存放在这个类中。
 * 优化的目的是调整位姿，位姿计算按复杂程度由低到高一共包含以下四种：
 * 1）当前帧位姿计算
 * 2）闭环检测时两帧之间相对位姿计算
 * 3）局部地图关键帧位姿和地图点位置调整
 * 4）全局地图关键帧位姿和地图点位置调整
 * 
 * 下面列出位姿计算和优化函数之间的对应关系：
 * 1）当前帧位姿计算： PoseOptimization
 *    每个帧可见多个地图点, 可以建立多个边连接, 构成图进行优化. 
 *    只对这一帧的SE3位姿进行优化，不优化地图点坐标。
 * 
 * 2）闭环检测时两帧之间相对位姿计算： OptimizeSim3
 *    优化两帧之间的位姿变换, 因为两帧之间可以看到多个相同的地图点, 可以构成一个超定方程组, 
 *    可以最小化误差优化。优化帧间变化的 SIM3位姿 与地图的 VertexSBAPointXYZ 位姿
 * 
 * 3）局部地图调整： LocalBundleAdjustment
 *    在一个 CovisbilityMap 内进行优化. 在一定范围的 keyframe 中,可以看到同一块地图,
 *    即是 CovisbilityMap .
 *    连接 CovisbilityMap 内的每一个 MapPoint 点与可见它的所有 keyframe ,
 *    放在一起进行优化.
 *    这个优化是双向的, 既优化地图点的 VertexSBAPointXYZ (MapPoints) 位姿, 
 *    又优化 frame 的 SE3 位姿.
 * 
 * 4）全局地图调整（简化版）： OptimizeEssentialGraph
 *    加入 Loop Closure 的考虑, Covisbility 图中的 keyframe 相互连起来, 
 *    Keyframe 之间有前后相连, 于是不同的 Covisbility 图也可以联系起来. 
 *    这是一个大范围的优化, 主要是加入了 **Loop Closure** 约束. 
 *    优化的是 Camera 的 SIM3 位姿.
 * 
 * 5）全局地图调整（完整版）： GlobalBundleAdjustemnt
 *    最大范围的优化, 优化所有 Camera 的 SE3 位姿与地图点的 XYZ 位姿.
 */

namespace ORB_SLAM2
{


/**
 * 参数列表
 * GlobalBundleAdjustemnt(
 * Map* pMap, //地图
 * int nIterations, //迭代次数
 * bool* pbStopFlag, //是否强制暂停
 * const unsigned long nLoopKF, //关键帧的个数
 * const bool bRobust)//是否使用核函数
 * 
 * 图的结构
 * Vertex:
 * -g2o::VertexSE3Expmap() ，当前帧的 Tcw
 * -g2o::VertexSBAPointXYZ() ， MapPoint 的 世界坐标(XYZ)
 * Edge:
 * -g2o::EdgeSE3ProjectXYZ()， BaseBinaryEdge
 * +Vertex ：待优化当前帧的 Tcw
 * +Vertex ：待优化 MapPoint 的 mWorldPos
 * +measurement： MapPoint 在当前帧中的像素坐标 (u,v)
 * +InfoMatrix: invSigma2 (与特征点所在的尺度有关)
 * 
 * 具体流程
 * 1) 提取出所有的 keyframes (GetAllKeyFrames) 和所有的 地图点(GetAllMapPoints) .
 * 2) 把 keyframe 设置为图中的 节点
 * 3) 把每一个 地图点 设置为图中的 节点 , 然后对于每一个 地图点 ,
 *    找出 所有 能看到这个 地图点 的 keyframe .
 * 4) 对每一个 keyframe , 建立 边 。边的两端分别是 地图点的 SE3位姿 与当前 keyframe 的 SE 位姿，
 *    边的观测值 为该地图点在当前 keyframe 中的二维位置，
 *    信息矩阵(权重) 是观测值的 偏离程度 , 即 3D地图点 反投影回地图的 误差 。
 * 5) 构图完成, 进行优化.
 * 6) 把优化后的 地图点 和 keyframe 位姿 全部放回原本的地图中.
 */
void Optimizer::GlobalBundleAdjustemnt(Map* pMap, int nIterations, 
    bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust) {
    vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    vector<MapPoint*> vpMP = pMap->GetAllMapPoints();
    BundleAdjustment(vpKFs, vpMP, nIterations, pbStopFlag, nLoopKF, bRobust);
}


void Optimizer::BundleAdjustment(const vector<KeyFrame *> &vpKFs, 
    const vector<MapPoint *> &vpMP, int nIterations, bool* pbStopFlag, 
    const unsigned long nLoopKF, const bool bRobust) {
    
    vector<bool> vbNotIncludedMP;
    vbNotIncludedMP.resize(vpMP.size());

    // 初始化g2o优化器
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    long unsigned int maxKFid = 0;

    // Set KeyFrame vertices
    // 向优化器添加顶点
    // 添加关键帧位姿顶点，所有的关键帧
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKF->GetPose()));
        vSE3->setId(pKF->mnId);
        vSE3->setFixed(pKF->mnId==0);
        optimizer.addVertex(vSE3);
        if(pKF->mnId>maxKFid)
            maxKFid=pKF->mnId;
    }

    const float thHuber2D = sqrt(5.99);
    const float thHuber3D = sqrt(7.815);

    // Set MapPoint vertices
    // 添加 MapPoints 顶点， 所有的地图点
    for(size_t i=0; i<vpMP.size(); i++)
    {
        MapPoint* pMP = vpMP[i];
        if(pMP->isBad())
            continue;
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));

        // 计算顶点编号，让 MapPoint 的顶点编号在 KeyFrame 添加完之后的编号基础上接着增加
        const int id = pMP->mnId+maxKFid+1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        const map<KeyFrame*,size_t> observations = pMP->GetObservations();

        int nEdges = 0;
        // SET EDGES
        // 优化器添加投影边，每一个 MapPoint 都要执行一次遍历，添加遍历到的所有边
        for(map<KeyFrame*,size_t>::const_iterator mit=observations.begin(); 
            mit!=observations.end(); mit++) {

            KeyFrame* pKF = mit->first;
            if(pKF->isBad() || pKF->mnId>maxKFid)
                continue;

            nEdges++;

            // mvKeysUn 中存放的是 校正后的特征点
            const cv::KeyPoint &kpUn = pKF->mvKeysUn[mit->second];

            // mvuRight 里面默认值为-1,在双目和RGBD相机时会被赋值
            // 所以此处“<0”的判断条件代表只有左目观测，相反，则是双目均可观测，
            // 此处双目包括RGBD虚拟出的右目
            // 两者的区别在于观测(measurement)不同，
            // 前者是 kpUn.pt.x, kpUn.pt.y 组成的二维向量，
            // 后者是 kpUn.pt.x, kpUn.pt.y, kp_ur 组成的三维向量
            if(pKF->mvuRight[mit->second]<0)
            {
                Eigen::Matrix<double,2,1> obs;
                obs << kpUn.pt.x, kpUn.pt.y;

                g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();

                e->setVertex(0, 
                    dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, 
                    dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
                e->setMeasurement(obs);
                // 根据特征点在金字塔中的尺度等级，设置不同的信息矩阵
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                // 如果需要，则添加核函数
                if(bRobust)
                {
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber2D);
                }

                e->fx = pKF->fx;
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;

                optimizer.addEdge(e);
            }
            else
            {
                Eigen::Matrix<double,3,1> obs;
                const float kp_ur = pKF->mvuRight[mit->second];
                obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
                e->setMeasurement(obs);
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                e->setInformation(Info);

                if(bRobust)
                {
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber3D);
                }

                e->fx = pKF->fx;
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;
                e->bf = pKF->mbf;

                optimizer.addEdge(e);
            }
        }

        if(nEdges==0)
        {
            optimizer.removeVertex(vPoint);
            vbNotIncludedMP[i]=true;
        }
        else
        {
            vbNotIncludedMP[i]=false;
        }
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(nIterations);

    // Recover optimized data

    //Keyframes
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        if(nLoopKF==0)
        {
            pKF->SetPose(Converter::toCvMat(SE3quat));
        }
        else
        {
            pKF->mTcwGBA.create(4,4,CV_32F);
            Converter::toCvMat(SE3quat).copyTo(pKF->mTcwGBA);
            pKF->mnBAGlobalForKF = nLoopKF;
        }
    }

    //Points
    for(size_t i=0; i<vpMP.size(); i++)
    {
        if(vbNotIncludedMP[i])
            continue;

        MapPoint* pMP = vpMP[i];

        if(pMP->isBad())
            continue;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));

        if(nLoopKF==0)
        {
            pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
            pMP->UpdateNormalAndDepth();
        }
        else
        {
            pMP->mPosGBA.create(3,1,CV_32F);
            Converter::toCvMat(vPoint->estimate()).copyTo(pMP->mPosGBA);
            pMP->mnBAGlobalForKF = nLoopKF;
        }
    }

}


/**
 * 当前帧位姿计算
 * 
 * 1) 直接把当前 keyframe 设为地图中的 节点
 * 2) 找出所有在当前 keyframe 中可见的的三维地图点, 对每一个地图点, 建立边。
 *    边的两端分别是 keyframe 的 位姿 与当前地图点为位姿，
 *    边的观测值为该地图点在当前 keyframe 中的 二维位置 ，
 *    信息矩阵(权重)是观测值的偏离程度, 即3D地图点反投影回地图的误差。
 * 3) 构图完成,进行优化.
 * 4) 把优化后的keyframe位姿放回去.
 * 
 * 
 * Vertex:
 * - g2o::VertexSE3Expmap()，当前帧的Tcw
 * 
 * Edge:
 * - g2o::EdgeSE3ProjectXYZOnlyPose() ， BaseUnaryEdge
 * + Vertex ：待优化当前帧的 Tcw
 * + measurement ：MapPoint在当前帧中的二维位置 (u,v)
 * + InfoMatrix : invSigma2 (与特征点所在的尺度有关)
 * - g2o::EdgeStereoSE3ProjectXYZOnlyPose()， BaseUnaryEdge
 * + Vertex：待优化当前帧的 Tcw
 * + measurement ：MapPoint在当前帧中的二维位置 (ul,v,ur)
 * + InfoMatrix : invSigma2 (与特征点所在的尺度有关)
 */
int Optimizer::PoseOptimization(Frame *pFrame)
{
    // 构造g2o优化器
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    linearSolver = 
        new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = 
        new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    int nInitialCorrespondences=0;

    // Set Frame vertex
    // 添加顶点：待优化当前帧的 Tcw
    g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
    vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
    vSE3->setId(0);
    vSE3->setFixed(false);
    optimizer.addVertex(vSE3);

    // Set MapPoint vertices
    const int N = pFrame->N;

    vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono;
    vector<size_t> vnIndexEdgeMono;
    vpEdgesMono.reserve(N);
    vnIndexEdgeMono.reserve(N);

    vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose*> vpEdgesStereo;
    vector<size_t> vnIndexEdgeStereo;
    vpEdgesStereo.reserve(N);
    vnIndexEdgeStereo.reserve(N);

    const float deltaMono = sqrt(5.991);
    const float deltaStereo = sqrt(7.815);


    {
    unique_lock<mutex> lock(MapPoint::mGlobalMutex);

    // 遍历 pFrame 帧的所有特征点，添加g2o边
    for(int i=0; i<N; i++)
    {
        MapPoint* pMP = pFrame->mvpMapPoints[i];
        if(pMP)
        {
            // Monocular observation
            // 此处 "<0"即代表单目
            if(pFrame->mvuRight[i]<0)
            {
                nInitialCorrespondences++;
                //先将这个特征点设置为 不是 Outlier ，也就是初始化
                pFrame->mvbOutlier[i] = false;

                Eigen::Matrix<double,2,1> obs;
                const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                obs << kpUn.pt.x, kpUn.pt.y; // [u, v]

                g2o::EdgeSE3ProjectXYZOnlyPose* e = 
                    new g2o::EdgeSE3ProjectXYZOnlyPose();

                e->setVertex(0, 
                    dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0))); // vSE3
                e->setMeasurement(obs); // [u, v]
                const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                e->setInformation(Eigen::Matrix2d::Identity()*invSigma2); // Info matrix 2d

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(deltaMono);

                e->fx = pFrame->fx;
                e->fy = pFrame->fy;
                e->cx = pFrame->cx;
                e->cy = pFrame->cy;
                
                // 在 globalBA 中， MapPoint 是以顶点形式出现的，此处由于只优化帧的位姿，
                // 不优化 MapPoint 位置，所以 MapPoint 就以下面这种形式存在
                cv::Mat Xw = pMP->GetWorldPos();
                e->Xw[0] = Xw.at<float>(0);
                e->Xw[1] = Xw.at<float>(1);
                e->Xw[2] = Xw.at<float>(2);

                optimizer.addEdge(e);

                vpEdgesMono.push_back(e);
                vnIndexEdgeMono.push_back(i);
            }
            // 双目的观测，包括RGBD虚拟出的双目，与单目的区别是观测多了一个 kp_ur ，由二维变成了三维
            else  // Stereo observation
            {
                nInitialCorrespondences++;
                pFrame->mvbOutlier[i] = false;

                //SET EDGE
                Eigen::Matrix<double,3,1> obs;
                const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                const float &kp_ur = pFrame->mvuRight[i];
                obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = 
                    new g2o::EdgeStereoSE3ProjectXYZOnlyPose();

                e->setVertex(0, 
                    dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                e->setMeasurement(obs);
                const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2; // Info matrix 3d
                e->setInformation(Info);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(deltaStereo);

                e->fx = pFrame->fx;
                e->fy = pFrame->fy;
                e->cx = pFrame->cx;
                e->cy = pFrame->cy;
                e->bf = pFrame->mbf; // mono doesn't have this
                cv::Mat Xw = pMP->GetWorldPos();
                e->Xw[0] = Xw.at<float>(0);
                e->Xw[1] = Xw.at<float>(1);
                e->Xw[2] = Xw.at<float>(2);

                optimizer.addEdge(e);

                vpEdgesStereo.push_back(e);
                vnIndexEdgeStereo.push_back(i);
            }
        }

    }
    }

    if(nInitialCorrespondences<3)
        return 0;

    // We perform 4 optimizations, after each optimization 
    // we classify observation as inlier/outlier
    // At the next optimization, outliers are not included, 
    // but at the end they can be classified as inliers again.
    // 开始优化，总共优化四次，
    // 每次优化后，将 **观测** 分为 outlier 和 inlier， 
    // outlier不参与下次优化
    // 由于每次优化后是对所有的观测进行 outlier 和 inlier 判别，
    // 因此之前被判别为outlier有可能变成inlier，反之亦然
    const float chi2Mono[4]={5.991,5.991,5.991,5.991};
    const float chi2Stereo[4]={7.815,7.815,7.815, 7.815};
    const int its[4]={10,10,10,10};    

    int nBad=0;
    for(size_t it=0; it<4; it++)
    {

        vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
        // 只对level为0的边进行优化，此处 0 代表 inlier ，1 代表 outlier
        optimizer.initializeOptimization(0);
        optimizer.optimize(its[it]);

        nBad=0;
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
        {
            g2o::EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMono[i];

            const size_t idx = vnIndexEdgeMono[i];

            if(pFrame->mvbOutlier[idx])
            {
                e->computeError(); // NOTE g2o只会计算active edge的误差
            }

            const float chi2 = e->chi2();

            if(chi2>chi2Mono[it])
            {                
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1); // 设置为 outlier
                nBad++;
            }
            else
            {
                pFrame->mvbOutlier[idx]=false; 
                e->setLevel(0); // 设置为inlier
            }

            if(it==2)
                e->setRobustKernel(0); // 前两次优化需要 RobustKernel , 其余的不需要
        }

        for(size_t i=0, iend=vpEdgesStereo.size(); i<iend; i++)
        {
            g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = vpEdgesStereo[i];

            const size_t idx = vnIndexEdgeStereo[i];

            if(pFrame->mvbOutlier[idx])
            {
                e->computeError(); // NOTE g2o 只会计算 active edge 的误差
            }

            const float chi2 = e->chi2();

            if(chi2>chi2Stereo[it])
            {
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1); // 设置为outlier
                nBad++;
            }
            else
            {                
                e->setLevel(0); // 设置为inlier
                pFrame->mvbOutlier[idx]=false;
            }

            if(it==2)
                e->setRobustKernel(0); // 前两次优化需要 RobustKernel , 其余的不需要
        }

        if(optimizer.edges().size()<10)
            break;
    }    

    // Recover optimized pose and return number of inliers
    g2o::VertexSE3Expmap* vSE3_recov = 
        static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
    g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
    cv::Mat pose = Converter::toCvMat(SE3quat_recov);
    pFrame->SetPose(pose); // 把优化后的位姿赋给当前帧

    return nInitialCorrespondences-nBad; // inliers 个数
}


/**
 * KeyFrame *pKF, // 当前关键帧
 * bool* pbStopFlag, // 是否强制暂停
 * Map* pMap //地图
 * 
 * 具体流程
 * 1) 找到 Local Keyframe , 即那些共享 CovisbilityMap 的 Keyframes . 存入 lLocalKeyFrames .
 * 2) 找到所有 Local Keyframes 都能看到的地图点, 其实就是 CovisbilityMap 的地图点.
 *    存入 lLocalMapPoints .
 * 3) 再找出能看到上面的地图点, 却不在 Local Keyframe 范围内的 keyframe . 
 *    存入 lFixedCameras .
 * 4) 把上面的 Local Keyframe , Map Point, FixedCamera 都设置为图节点.
 * 5) 对于 lLocalMapPoints 中的每一个地图点及能看到它的所有 keyframes , 建立边。
 *    边的两端分别是 keyframe 的 位姿 与当前地图点为位姿，边的 观测值 为该地图点在当前 keyframe 中 
 *    的二维位置，信息矩阵(权重)是观测值的偏离程度, 即3D地图点反投影回地图的 误差 .
 * 6) 去除掉一些不符合标准的边.
 * 7) 把优化后地图点和keyframe位姿放回去.
 * 
 * Vertex:
 * - g2o::VertexSE3Expmap()，局部图，当前关键帧、与当前关键帧相连的关键帧的 位姿
 * - g2o::VertexSE3Expmap()，即能观测到局部地图点的关键帧（并且不属于 LocalKeyFrames ）的 位姿 ，
 *                           在优化中这些关键帧的位姿不变
 * - g2o::VertexSBAPointXYZ()，局部地图点，即局部图能观测到的所有地图点的位置
 * 
 * Edge:
 * - g2o::EdgeSE3ProjectXYZ()， BaseBinaryEdge
 * + Vertex：关键帧的 Tcw ，MapPoint 的 Pw
 * + measurement：地图点在关键帧中的二维位置 (u,v)
 * + InfoMatrix:  invSigma2
 * - g2o::EdgeStereoSE3ProjectXYZ()， BaseBinaryEdge
 * + Vertex：关键帧的 Tcw ， MapPoint 的 Pw
 * + measurement：地图点在关键帧中的二维位置 (ul,v,ur)
 * + InfoMatrix: invSigma2
 */
void Optimizer::LocalBundleAdjustment(KeyFrame *pKF, bool* pbStopFlag, Map* pMap)
{    
    // Local KeyFrames: First Breath Search from Current Keyframe
    list<KeyFrame*> lLocalKeyFrames; // in the CovisbilityMap
    // 将当前 关键帧 加入 lLocalKeyFrames
    lLocalKeyFrames.push_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId;

    // 找到关键帧一级 (in CovisbilityMap) 连接的关键帧，加入lLocalKeyFrames中
    const vector<KeyFrame*> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
    for(int i=0, iend=vNeighKFs.size(); i<iend; i++)
    {
        KeyFrame* pKFi = vNeighKFs[i];
        pKFi->mnBALocalForKF = pKF->mnId;
        if(!pKFi->isBad())
            lLocalKeyFrames.push_back(pKFi);
    }

    // Local MapPoints seen in Local KeyFrames (CovisibleKeyFrames)
    list<MapPoint*> lLocalMapPoints;
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin() , 
        lend=lLocalKeyFrames.end(); lit!=lend; lit++) {
        vector<MapPoint*> vpMPs = (*lit)->GetMapPointMatches();
        for(vector<MapPoint*>::iterator vit=vpMPs.begin(), vend=vpMPs.end(); vit!=vend; vit++)
        {
            MapPoint* pMP = *vit;
            if(pMP)
                if(!pMP->isBad())
                    if(pMP->mnBALocalForKF!=pKF->mnId)
                    {
                        lLocalMapPoints.push_back(pMP);
                        pMP->mnBALocalForKF=pKF->mnId;
                    }
        }
    }

    // Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
    // 得到能被局部 MapPoints 观测到，但不属于局部关键帧的关键帧，这些关键帧在局部 BA 优化时不优化
    list<KeyFrame*> lFixedCameras;
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        map<KeyFrame*,size_t> observations = (*lit)->GetObservations();
        for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if(pKFi->mnBALocalForKF!=pKF->mnId && pKFi->mnBAFixedForKF!=pKF->mnId)
            {                
                pKFi->mnBAFixedForKF=pKF->mnId;
                if(!pKFi->isBad())
                    lFixedCameras.push_back(pKFi);
            }
        }
    }


    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    unsigned long maxKFid = 0;


    // Set Local KeyFrame vertices
    // 把局部地图中的关键帧加入 顶点
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), 
        lend=lLocalKeyFrames.end(); lit!=lend; lit++) {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(pKFi->mnId==0);
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid)
            maxKFid=pKFi->mnId;
    }


    // Set Fixed KeyFrame vertices
    // 把能被局部 MapPoints 观测到，但不属于 局部地图的 关键帧 加入 顶点
    for(list<KeyFrame*>::iterator lit=lFixedCameras.begin(), 
        lend=lFixedCameras.end(); lit!=lend; lit++) {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(true); // 因为这些顶点不需要优化，所以设值为fixed
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid)
            maxKFid=pKFi->mnId;
    }


    // Set MapPoint vertices
    const int nExpectedSize = (lLocalKeyFrames.size()+lFixedCameras.size())
                              *lLocalMapPoints.size();

    vector<g2o::EdgeSE3ProjectXYZ*> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);

    vector<g2o::EdgeStereoSE3ProjectXYZ*> vpEdgesStereo;
    vpEdgesStereo.reserve(nExpectedSize);

    vector<KeyFrame*> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);

    vector<MapPoint*> vpMapPointEdgeStereo;
    vpMapPointEdgeStereo.reserve(nExpectedSize);

    const float thHuberMono = sqrt(5.991);
    const float thHuberStereo = sqrt(7.815);

    // 把所有 MapPoint 加入顶点，并添加与 MapPoint 相连的边
    for(list<MapPoint*>::iterator 
        lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++) {

        // 把所有 MapPoint 加入 顶点
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
        // 计算顶点编号，让 MapPoint 的顶点编号在 KeyFrame 
        // 添加完之后的编号基础上接着增加
        int id = pMP->mnId+maxKFid+1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        const map<KeyFrame*,size_t> observations = pMP->GetObservations();

        // Set edges
        // 添加与 MapPoint 相连的边
        for(map<KeyFrame*,size_t>::const_iterator mit=observations.begin(), 
            mend=observations.end(); mit!=mend; mit++) {

            KeyFrame* pKFi = mit->first;

            if(!pKFi->isBad())
            {                
                const cv::KeyPoint &kpUn = pKFi->mvKeysUn[mit->second];

                // Monocular observation
                if(pKFi->mvuRight[mit->second]<0)
                {
                    Eigen::Matrix<double,2,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();

                    e->setVertex(0, 
                        dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, 
                        dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;

                    optimizer.addEdge(e);
                    vpEdgesMono.push_back(e);
                    vpEdgeKFMono.push_back(pKFi);
                    vpMapPointEdgeMono.push_back(pMP);
                }
                else // Stereo observation
                {
                    Eigen::Matrix<double,3,1> obs;
                    const float kp_ur = pKFi->mvuRight[mit->second];
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                    e->setInformation(Info);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;
                    e->bf = pKFi->mbf;

                    optimizer.addEdge(e);
                    vpEdgesStereo.push_back(e);
                    vpEdgeKFStereo.push_back(pKFi);
                    vpMapPointEdgeStereo.push_back(pMP);
                }
            }
        }
    }

    if(pbStopFlag)
        if(*pbStopFlag)
            return;

    optimizer.initializeOptimization();
    optimizer.optimize(5);

    bool bDoMore= true;

    if(pbStopFlag)
        if(*pbStopFlag)
            bDoMore = false;

    if(bDoMore)
    {

    // 检测outlier，并设置下次不优化
    // Check inlier observations
    for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
    {
        g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
        MapPoint* pMP = vpMapPointEdgeMono[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>5.991 || !e->isDepthPositive())
        {
            e->setLevel(1); // 不优化
        }

        e->setRobustKernel(0); // 不使用核函数
    }

    for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
    {
        g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
        MapPoint* pMP = vpMapPointEdgeStereo[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>7.815 || !e->isDepthPositive())
        {
            e->setLevel(1);
        }

        e->setRobustKernel(0);
    }

    // Optimize again without the outliers
    optimizer.initializeOptimization(0);
    optimizer.optimize(10);
    }

    vector<pair<KeyFrame*,MapPoint*> > vToErase;
    vToErase.reserve(vpEdgesMono.size()+vpEdgesStereo.size());

    // Check inlier observations
    // 优化后重新计算 误差 ，剔除 连接误差 比较大的关键帧和地图点
    for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
    {
        g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
        MapPoint* pMP = vpMapPointEdgeMono[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>5.991 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFMono[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }

    for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
    {
        g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
        MapPoint* pMP = vpMapPointEdgeStereo[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>7.815 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFStereo[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }

    // Get Map Mutex
    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    // 偏差比较大，在关键帧中剔除对该地图点的观测
    // 在地图点中剔除对该关键帧的观测
    if(!vToErase.empty())
    {
        for(size_t i=0;i<vToErase.size();i++)
        {
            KeyFrame* pKFi = vToErase[i].first;
            MapPoint* pMPi = vToErase[i].second;
            pKFi->EraseMapPointMatch(pMPi);
            pMPi->EraseObservation(pKFi);
        }
    }

    // Recover optimized data
    // 优化后更新 关键帧 位姿 以及 MapPoints 的 位置 、平均观测 方向 等属性

    //Keyframes
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), 
        lend=lLocalKeyFrames.end(); lit!=lend; lit++) {
        KeyFrame* pKF = *lit;
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        pKF->SetPose(Converter::toCvMat(SE3quat));
    }

    //Points
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), 
        lend=lLocalMapPoints.end(); lit!=lend; lit++) {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));
        pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
        pMP->UpdateNormalAndDepth();
    }
}


/**
 * 参数列表
 * OptimizeEssentialGraph(
 * Map* pMap, // 地图
 * KeyFrame* pLoopKF, // 闭环匹配上的帧
 * KeyFrame* pCurKF,// 当前帧
 * const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,// 未经过Sim3调整的位姿
 * const LoopClosing::KeyFrameAndPose &CorrectedSim3,// 经过Sim3调整的位姿
 * const map<KeyFrame *, set<KeyFrame *> > &LoopConnections, // 闭环连接关系
 * const bool &bFixScale)//是否优化尺度，单目需要优化，双目不需要优化
 * 
 * 图的结构
 * Vertex:
 * - g2o::VertexSim3Expmap，Essential graph 中关键帧的 位姿
 * Edge:
 * - g2o::EdgeSim3()， BaseBinaryEdge
 * + Vertex： 关键帧的Tcw，MapPoint的Pw
 * + measurement： 经过 CorrectLoop 函数 Sim3传播校正后的位姿
 * + InfoMatrix:  单位矩阵
 * 
 * 具体流程
 * 1) 首先获取所有 keyframes 和 地图点
 * 2) 把 keyframe 都设为图 节点 , 剔除不好的 keyframe . 这里的frame位姿为 Sim3 .
 * 3) 添加边
 * <1> Loop edge: LoopConnections 是一个 Map, Map 中第一个元素是有 loop 的 keyframe , 
 *     第二个元素是与第一个元素形成 loop 的 keyframe 集合. 给它们全部添加边进行连接。
 *     边的观测值是后一帧的 SIM3位姿 乘以 前一帧的 SIM3位姿 的逆.
 * <2> Normal edge: 遍历所有的keyframe
 *     - 找到 当前 keyframe 的 parent keyframe, 建立边连接. 
 *       边的观测值为 parent keyframe 的位姿 乘以 keyframe 位姿 的 逆. 
 *       信息矩阵为单位矩阵.
 *     - 找到与当前 keyframe 形成 Loop 的所有 keyframe , 
 *       如果找到成 Loop 的 keyframe 在 当前keyframe 之前, 则在两个keyframe之间添加一个 边连接 .
 *       观测值为 Loop keyframe 的位姿 乘以 keyframe 位姿 的 逆. 
 *       信息矩阵为单位矩阵.
 *     - 找到当前 keyframe 的 covisibility graph 中的每一个 keyframe , 建立边连接.
 *       观测值为 covisibility graph keyframe 位姿 乘以 keyframe 位姿 的逆.
 *       信息矩阵为单位矩阵.
 * 4) 构图完成, 进行优化.
 * 5) 更新EssentialGraph中的所有位姿.
 */
void Optimizer::OptimizeEssentialGraph(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
                            const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
                            const LoopClosing::KeyFrameAndPose &CorrectedSim3,
                            const map<KeyFrame *, set<KeyFrame *> > &LoopConnections, 
                            const bool &bFixScale)
{
    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    // 这表明误差变量为 7维 ，误差项为 3维
    g2o::BlockSolver_7_3::LinearSolverType * linearSolver =
           new g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>();
    g2o::BlockSolver_7_3 * solver_ptr= new g2o::BlockSolver_7_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver =
        new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    solver->setUserLambdaInit(1e-16);
    optimizer.setAlgorithm(solver);

    const vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    const vector<MapPoint*> vpMPs = pMap->GetAllMapPoints();

    const unsigned int nMaxKFid = pMap->GetMaxKFid();

    // 经过 Sim3 传播调整，未经过优化的 keyframe 的 pose
    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vScw(nMaxKFid+1);
    // 经过 Sim3 传播调整，经过优化的 keyframe 的 pose
    vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vCorrectedSwc(nMaxKFid+1);
    vector<g2o::VertexSim3Expmap*> vpVertices(nMaxKFid+1);

    const int minFeat = 100;

    // Set KeyFrame vertices
    // 将地图中所有 keyframe 的 pose 作为 顶点 添加到优化器
    // 尽可能使用经过 Sim3 调整的 位姿
    for(size_t i=0, iend=vpKFs.size(); i<iend;i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
        g2o::VertexSim3Expmap* VSim3 = new g2o::VertexSim3Expmap();

        const int nIDi = pKF->mnId;

        LoopClosing::KeyFrameAndPose::const_iterator it = CorrectedSim3.find(pKF);

        // 如果该关键帧在闭环时通过Sim3传播调整过，用校正后的位姿
        if(it!=CorrectedSim3.end())
        {
            vScw[nIDi] = it->second;
            VSim3->setEstimate(it->second);
        }
        else
        // 如果没有通过Sim3传播调整过，用自身的位姿
        {
            Eigen::Matrix<double,3,3> Rcw = Converter::toMatrix3d(pKF->GetRotation());
            Eigen::Matrix<double,3,1> tcw = Converter::toVector3d(pKF->GetTranslation());
            g2o::Sim3 Siw(Rcw,tcw,1.0);
            vScw[nIDi] = Siw;
            VSim3->setEstimate(Siw);
        }

        // 闭环匹配上的帧 不进行 位姿优化
        // vpKFs: all keyframes
        if(pKF==pLoopKF)
            VSim3->setFixed(true);

        VSim3->setId(nIDi);
        VSim3->setMarginalized(false);
        VSim3->_fix_scale = bFixScale;

        optimizer.addVertex(VSim3);

        vpVertices[nIDi]=VSim3;
    }

    // 在 g2o 中已经形成误差边的两个顶点，firstid 数较小的顶点
    set<pair<long unsigned int,long unsigned int> > sInsertedEdges;

    const Eigen::Matrix<double,7,7> matLambda = Eigen::Matrix<double,7,7>::Identity();

    // Set Loop edges
    // 添加边： LoopConnections 是闭环时 因为地图点调整 而出现的 新关键帧 连接关系
    for(map<KeyFrame *, set<KeyFrame *> >::const_iterator mit = LoopConnections.begin(), 
        mend=LoopConnections.end(); mit!=mend; mit++) {
        
        KeyFrame* pKF = mit->first;
        const long unsigned int nIDi = pKF->mnId;
        const set<KeyFrame*> &spConnections = mit->second;
        const g2o::Sim3 Siw = vScw[nIDi];
        const g2o::Sim3 Swi = Siw.inverse();

        for(set<KeyFrame*>::const_iterator sit=spConnections.begin(), send=spConnections.end(); sit!=send; sit++)
        {
            const long unsigned int nIDj = (*sit)->mnId;
            if((nIDi!=pCurKF->mnId || nIDj!=pLoopKF->mnId) && pKF->GetWeight(*sit)<minFeat)
                continue;

            const g2o::Sim3 Sjw = vScw[nIDj];
            // 得到两个 pose 间的 Sim3变换
            const g2o::Sim3 Sji = Sjw * Swi;

            g2o::EdgeSim3* e = new g2o::EdgeSim3();
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            e->setMeasurement(Sji);

            e->information() = matLambda;

            optimizer.addEdge(e);

            sInsertedEdges.insert(make_pair(min(nIDi,nIDj),max(nIDi,nIDj)));
        }
    }

    // Set normal edges
    // 添加跟踪时形成的边、闭环匹配成功形成的边
    // 遍历 vpKFs ，将 vpKFs 和其在 spanningtree 中的父节点在g2o图中连接起来形成一条 误差边 ；
    // 将 vpKFs 和其形成闭环的帧在g2o图中连接起来形成一条 误差边
    for(size_t i=0, iend=vpKFs.size(); i<iend; i++)
    {
        KeyFrame* pKF = vpKFs[i];

        const int nIDi = pKF->mnId;

        g2o::Sim3 Swi;

        LoopClosing::KeyFrameAndPose::const_iterator iti = NonCorrectedSim3.find(pKF);

        if(iti!=NonCorrectedSim3.end())
            Swi = (iti->second).inverse();
        else
            Swi = vScw[nIDi].inverse();

        KeyFrame* pParentKF = pKF->GetParent();

        // Spanning tree edge
        // 只添加扩展树的边（有父关键帧）
        // 将 vpKFs 和其在 spanningtree 中的父节点在g2o图中连接起来形成一条 误差边 ；
        if(pParentKF)
        {
            int nIDj = pParentKF->mnId;

            g2o::Sim3 Sjw;

            LoopClosing::KeyFrameAndPose::const_iterator itj = NonCorrectedSim3.find(pParentKF);

            // 尽可能得到未经过 Sim3 传播调整的位姿
            if(itj!=NonCorrectedSim3.end())
                Sjw = itj->second;
            else
                Sjw = vScw[nIDj];

            // 得到两个 pose 间的 Sim3变换
            g2o::Sim3 Sji = Sjw * Swi;

            g2o::EdgeSim3* e = new g2o::EdgeSim3();
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            e->setMeasurement(Sji);

            e->information() = matLambda;
            optimizer.addEdge(e);
        }

        // Loop edges
        // 步添加在 CorrectLoop 函数 中 AddLoopEdge 函数添加的 闭环连接边（当前帧与闭环匹配帧之间的连接关系）
        // 使用经过 Sim3调整 前 关键帧之间的相对关系 作为 边
        // 将 vpKFs 和 其形成闭环的帧 在g2o图中连接起来形成一条 误差边
        const set<KeyFrame*> sLoopEdges = pKF->GetLoopEdges();
        for(set<KeyFrame*>::const_iterator sit=sLoopEdges.begin(), send=sLoopEdges.end(); sit!=send; sit++)
        {
            KeyFrame* pLKF = *sit;
            if(pLKF->mnId<pKF->mnId)
            {
                g2o::Sim3 Slw;

                LoopClosing::KeyFrameAndPose::const_iterator itl = NonCorrectedSim3.find(pLKF);

                // 尽可能得到未经过Sim3传播调整的位姿
                if(itl!=NonCorrectedSim3.end())
                    Slw = itl->second;
                else
                    Slw = vScw[pLKF->mnId];

                g2o::Sim3 Sli = Slw * Swi;
                g2o::EdgeSim3* el = new g2o::EdgeSim3();
                el->setVertex(1, 
                    dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pLKF->mnId)));
                el->setVertex(0, 
                    dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                el->setMeasurement(Sli);
                el->information() = matLambda;
                optimizer.addEdge(el);
            }
        }

        // Covisibility graph edges
        // 有很好共视关系的关键帧也作为边进行优化
        // 使用 经过Sim3调整前 关键帧 之间的相对关系作为 边
        // pKF 与在 Covisibility graph 中与pKF连接，且共视点超过 minFeat 的关键帧，
        // 形成一条 误差边 （如果之前没有添加过的话）        
        const vector<KeyFrame*> vpConnectedKFs = pKF->GetCovisiblesByWeight(minFeat);
        for(vector<KeyFrame*>::const_iterator vit=vpConnectedKFs.begin(); vit!=vpConnectedKFs.end(); vit++)
        {
            KeyFrame* pKFn = *vit;
            // 避免和前面的边添加重复
            if(pKFn && pKFn!=pParentKF && !pKF->hasChild(pKFn) && !sLoopEdges.count(pKFn))
            {
                if(!pKFn->isBad() && pKFn->mnId<pKF->mnId)
                {
                    // 为避免重复添加，先查找
                    if(sInsertedEdges.count(
                        make_pair(min(pKF->mnId,pKFn->mnId), max(pKF->mnId,pKFn->mnId))))
                        continue;

                    g2o::Sim3 Snw;

                    LoopClosing::KeyFrameAndPose::const_iterator itn = NonCorrectedSim3.find(pKFn);

                    // 尽可能得到未经过 Sim3 传播调整的 位姿
                    if(itn!=NonCorrectedSim3.end())
                        Snw = itn->second;
                    else
                        Snw = vScw[pKFn->mnId];

                    g2o::Sim3 Sni = Snw * Swi;

                    g2o::EdgeSim3* en = new g2o::EdgeSim3();
                    en->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFn->mnId)));
                    en->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
                    en->setMeasurement(Sni);
                    en->information() = matLambda;
                    optimizer.addEdge(en);
                }
            }
        }
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(20);

    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
    // 更新优化后的闭环检测位姿
    for(size_t i=0;i<vpKFs.size();i++)
    {
        KeyFrame* pKFi = vpKFs[i];

        const int nIDi = pKFi->mnId;

        g2o::VertexSim3Expmap* VSim3 = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(nIDi));
        g2o::Sim3 CorrectedSiw =  VSim3->estimate();
        vCorrectedSwc[nIDi]=CorrectedSiw.inverse();
        Eigen::Matrix3d eigR = CorrectedSiw.rotation().toRotationMatrix();
        Eigen::Vector3d eigt = CorrectedSiw.translation();
        double s = CorrectedSiw.scale();

        eigt *=(1./s); //[R t/s;0 1]

        cv::Mat Tiw = Converter::toCvSE3(eigR,eigt);

        pKFi->SetPose(Tiw);
    }

    // Correct points. Transform to "non-optimized" reference keyframe pose 
    // and transform back with optimized pose
    // 优化得到关键帧的位姿后，地图点根据参考帧优化前后的相对关系调整自己的位置
    for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
    {
        MapPoint* pMP = vpMPs[i];

        if(pMP->isBad())
            continue;

        int nIDr;
        // 该地图点经过 Sim3 调整过
        if(pMP->mnCorrectedByKF==pCurKF->mnId)
        {
            nIDr = pMP->mnCorrectedReference;
        }
        else
        {
            KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();
            nIDr = pRefKF->mnId;
        }

        // 得到地图点参考关键帧 优化 前 的位姿
        g2o::Sim3 Srw = vScw[nIDr];
        // 得到地图点参考关键帧 优化 后 的位姿
        g2o::Sim3 correctedSwr = vCorrectedSwc[nIDr];

        cv::Mat P3Dw = pMP->GetWorldPos();
        Eigen::Matrix<double,3,1> eigP3Dw = Converter::toVector3d(P3Dw);
        Eigen::Matrix<double,3,1> eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));

        cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
        pMP->SetWorldPos(cvCorrectedP3Dw);

        pMP->UpdateNormalAndDepth();
    }
}


/**
 * 闭环检测 时 两帧之间 相对位姿 计算
 * KeyFrame *pKF1, // 匹配的两帧中的 一帧
 * KeyFrame *pKF2, // 匹配的两帧中的 另一帧
 * vector<MapPoint *> &vpMatches1, // 共视的地图点
 * g2o::Sim3 &g2oS12, // 两个关键帧间的 Sim3 变换
 * const float th2, // 核函数阈值
 * const bool bFixScale) // 单目 进行 尺度优化，双目 不进行 尺度优化
 * 
 * 具体流程
 * 1) 把输入的 KF1 到 KF2 的位姿变换 SIM3 加入图中作为节点0.
 * 2) 找到 KF1 中对应的所有map点, 放在 vpMapPoints1 中. 
 *    vpMatches1 为输入的匹配地图点, 是在 KF2 中匹配上 map 点的对应集合.
 * 3) Point1 是 KF1 的 Rotation matrix*map point1的世界坐标 + KF1的Translation matrix. 
 *    Point2 是 KF2 的 Rotation matrix*map point2的世界坐标 + KF2的Translation matrix. 
 *    把 Point1 Point2 作为节点加入图中
 * 4) 在节点 0 与 Point1 ,  Point2 之间都建立边连接. 
 *    测量值分别是地图点反投影在图像上的 二维坐标 , 信息矩阵 为反投影的误差.
 * 5) 图构建完成, 进行优化
 * 6) 更新两帧间转换的 Sim3.
 * 
 * Vertex:
 * - g2o::VertexSim3Expmap()，两个关键帧的 位姿
 * - g2o::VertexSBAPointXYZ()，两个关键帧共有的 地图点
 * Edge:
 * - g2o::EdgeSim3ProjectXYZ()， BaseBinaryEdge
 * + Vertex ：关键帧的 Sim3 ， MapPoint 的 Pw
 * + measurement ： MapPoint 在关键帧中的 二维位置(u,v)
 * + InfoMatrix : invSigma2 (与特征点所在的尺度有关)
 * - g2o::EdgeInverseSim3ProjectXYZ()， BaseBinaryEdge
 * + Vertex ：关键帧的 Sim3 ， MapPoint的Pw
 * + measurement ： MapPoint 在关键帧中的 二维位置(u,v)
 * + InfoMatrix : invSigma2 (与特征点所在的尺度有关)
 */
int Optimizer::OptimizeSim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> 
    &vpMatches1, g2o::Sim3 &g2oS12, const float th2, const bool bFixScale) {
    g2o::SparseOptimizer optimizer;
    // 这表明 误差变量 和 误差项 的 维度 是 动态 的
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    // Calibration
    const cv::Mat &K1 = pKF1->mK;
    const cv::Mat &K2 = pKF2->mK;

    // Camera poses
    const cv::Mat R1w = pKF1->GetRotation();
    const cv::Mat t1w = pKF1->GetTranslation();
    const cv::Mat R2w = pKF2->GetRotation();
    const cv::Mat t2w = pKF2->GetTranslation();

    // Set Sim3 vertex
    // 添加 sim3 位姿顶点误差变量
    g2o::VertexSim3Expmap * vSim3 = new g2o::VertexSim3Expmap();    
    vSim3->_fix_scale=bFixScale;
    vSim3->setEstimate(g2oS12);
    vSim3->setId(0);
    vSim3->setFixed(false);

    // 将内参导入顶点
    vSim3->_principle_point1[0] = K1.at<float>(0,2);    // cx
    vSim3->_principle_point1[1] = K1.at<float>(1,2);    // cy
    vSim3->_focal_length1[0] = K1.at<float>(0,0);       // fx
    vSim3->_focal_length1[1] = K1.at<float>(1,1);       // fy

    vSim3->_principle_point2[0] = K2.at<float>(0,2);    // cx
    vSim3->_principle_point2[1] = K2.at<float>(1,2);    // cy
    vSim3->_focal_length2[0] = K2.at<float>(0,0);       // fx
    vSim3->_focal_length2[1] = K2.at<float>(1,1);       // fy
    
    optimizer.addVertex(vSim3);

    // Set MapPoint vertices
    const int N = vpMatches1.size();
    // 获得 pKF1 的所有 mappoint
    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    vector<g2o::EdgeSim3ProjectXYZ*> vpEdges12;
    vector<g2o::EdgeInverseSim3ProjectXYZ*> vpEdges21;
    vector<size_t> vnIndexEdge;

    vnIndexEdge.reserve(2*N);
    vpEdges12.reserve(2*N);
    vpEdges21.reserve(2*N);

    const float deltaHuber = sqrt(th2);

    int nCorrespondences = 0;
    // 将匹配转化为 归一化3d点 作为g2o的 顶点
    for(int i=0; i<N; i++)
    {
        if(!vpMatches1[i]) 
            continue;

        MapPoint* pMP1 = vpMapPoints1[i];
        MapPoint* pMP2 = vpMatches1[i]; // match id in pKF2

        const int id1 = 2*i+1;
        const int id2 = 2*(i+1);

        const int i2 = pMP2->GetIndexInKeyFrame(pKF2);

        if(pMP1 && pMP2)
        {
            if(!pMP1->isBad() && !pMP2->isBad() && i2>=0)
            {
                g2o::VertexSBAPointXYZ* vPoint1 = new g2o::VertexSBAPointXYZ();
                cv::Mat P3D1w = pMP1->GetWorldPos();
                cv::Mat P3D1c = R1w*P3D1w + t1w; // R*P+t
                vPoint1->setEstimate(Converter::toVector3d(P3D1c));
                vPoint1->setId(id1);
                vPoint1->setFixed(true);
                optimizer.addVertex(vPoint1);

                g2o::VertexSBAPointXYZ* vPoint2 = new g2o::VertexSBAPointXYZ();
                cv::Mat P3D2w = pMP2->GetWorldPos();
                cv::Mat P3D2c = R2w*P3D2w + t2w;
                vPoint2->setEstimate(Converter::toVector3d(P3D2c));
                vPoint2->setId(id2);
                vPoint2->setFixed(true);
                optimizer.addVertex(vPoint2);
            }
            else
                continue;
        }
        else
            continue;

        nCorrespondences++;

        // Set edge x1 = S12*X2
        // 添加 误差项 边
        Eigen::Matrix<double,2,1> obs1;
        const cv::KeyPoint &kpUn1 = pKF1->mvKeysUn[i];
        obs1 << kpUn1.pt.x, kpUn1.pt.y;

        g2o::EdgeSim3ProjectXYZ* e12 = new g2o::EdgeSim3ProjectXYZ();
        // 将 e12 边和 vertex(id2) 绑定
        e12->setVertex(0, 
            dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id2)));
        e12->setVertex(1, 
            dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        // 设定初始值
        e12->setMeasurement(obs1);
        const float &invSigmaSquare1 = pKF1->mvInvLevelSigma2[kpUn1.octave];
        e12->setInformation(Eigen::Matrix2d::Identity()*invSigmaSquare1);

        g2o::RobustKernelHuber* rk1 = new g2o::RobustKernelHuber;
        e12->setRobustKernel(rk1);
        rk1->setDelta(deltaHuber);
        optimizer.addEdge(e12);

        // Set edge x2 = S21*X1
        Eigen::Matrix<double,2,1> obs2;
        const cv::KeyPoint &kpUn2 = pKF2->mvKeysUn[i2];
        obs2 << kpUn2.pt.x, kpUn2.pt.y;

        // 注意这个的边类型和上面不一样
        g2o::EdgeInverseSim3ProjectXYZ* e21 = new g2o::EdgeInverseSim3ProjectXYZ();

        e21->setVertex(0, 
            dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id1)));
        e21->setVertex(1, 
            dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        e21->setMeasurement(obs2);
        float invSigmaSquare2 = pKF2->mvInvLevelSigma2[kpUn2.octave];
        e21->setInformation(Eigen::Matrix2d::Identity()*invSigmaSquare2);

        g2o::RobustKernelHuber* rk2 = new g2o::RobustKernelHuber;
        e21->setRobustKernel(rk2);
        rk2->setDelta(deltaHuber);
        optimizer.addEdge(e21);

        vpEdges12.push_back(e12);
        vpEdges21.push_back(e21);
        vnIndexEdge.push_back(i);
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(5);

    // Check inliers
    // 把不是 inliner 的边剔除出去
    int nBad=0;
    for(size_t i=0; i<vpEdges12.size();i++)
    {
        g2o::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
        g2o::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
        if(!e12 || !e21)
            continue;

        if(e12->chi2()>th2 || e21->chi2()>th2)
        {
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx]=static_cast<MapPoint*>(NULL);
            optimizer.removeEdge(e12);
            optimizer.removeEdge(e21);
            vpEdges12[i]=static_cast<g2o::EdgeSim3ProjectXYZ*>(NULL);
            vpEdges21[i]=static_cast<g2o::EdgeInverseSim3ProjectXYZ*>(NULL);
            nBad++;
        }
    }

    int nMoreIterations;
    if(nBad>0)
        nMoreIterations=10;
    else
        nMoreIterations=5;

    if(nCorrespondences-nBad<10)
        return 0;

    // Optimize again only with inliers
    optimizer.initializeOptimization();
    optimizer.optimize(nMoreIterations);

    int nIn = 0;
    // 看哪些匹配是 inliner
    for(size_t i=0; i<vpEdges12.size();i++)
    {
        g2o::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
        g2o::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
        if(!e12 || !e21)
            continue;

        if(e12->chi2()>th2 || e21->chi2()>th2)
        {
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx]=static_cast<MapPoint*>(NULL);
        }
        else
            nIn++;
    }

    // Recover optimized Sim3
    g2o::VertexSim3Expmap* vSim3_recov = 
        static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(0));
    g2oS12= vSim3_recov->estimate();

    return nIn;
}


} //namespace ORB_SLAM
