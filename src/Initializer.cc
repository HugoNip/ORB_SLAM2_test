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

#include "Initializer.h"

#include "Thirdparty/DBoW2/DUtils/Random.h"

#include "Optimizer.h"
#include "ORBmatcher.h"

#include<thread>

namespace ORB_SLAM2
{

/**
 * 该函数的作用是 设置参考帧 ，并 设置初始化方法的参数
 * @param sigma         计算矩阵得分时候所用的参数
 * @param iterations    RANSAC迭代次数
 */
Initializer::Initializer(const Frame &ReferenceFrame, float sigma, int iterations)
{
    mK = ReferenceFrame.mK.clone();

    mvKeys1 = ReferenceFrame.mvKeysUn;

    mSigma = sigma;
    mSigma2 = sigma*sigma;
    mMaxIterations = iterations;
}


/**
 * @brief 
 * 这个函数包含了整个初始化的全部流程，主要包括以下步骤：
 * 1）**重新组织**特征点对。其实就是重新弄了一下数据结构，把匹配的点对序号放在一起，方便后面使用
 * 2）特征点对**分组**。这一步主要是为了给 RANSAC 使用，对特征点对按照 RANSAC循环次数 随机分组
 * 3）两个线程同时计算**单应性矩阵H**和**本质矩阵F**
 * 4）根据 三角化 成功点数 来判断是 **选** 单应性矩阵H 和 本质矩阵F 中的哪一个
 * 5）根据矩阵，使用 SFM方法 **恢复 R 和 t**
 * 
 * // Reference Frame: 1, Current Frame: 2
 * 
 * @param CurrentFrame      当前帧
 * @param vMatches12        ORB计算的初步匹配结果
 * @param R21               输出的旋转矩阵, 1: camera1 (ReferenceFrame), 2: camera2 (CurrentFrame) -> return
 * @param t21               输出的平移向量, Reference Frame: 1, Current Frame: 2    -> return
 * @param vP3D              其大小为vKeys1大小，表示三角化**重投影成功**的匹配点的3d点在相机1下的坐标, vector Projection 3D in camera 1
 * @param vbTriangulated    其大小为vKeys1大小，表示初始化成功后，特征点中三角化**投影成功**的情况, vector bool Triangulated
 * 
 * Tracking.cc
 * if(mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated)) // Rcw, tcw
 * // 初始化时 得到的 特征点匹配，大小是 mInitialFrame 的特征点数量，其值是 当前帧特征点 序号(idx)
 * std::vector<int> mvIniMatches;
 */
bool Initializer::Initialize(const Frame &CurrentFrame, const vector<int> &vMatches12, cv::Mat &R21, cv::Mat &t21,
                             vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated)
{
    // Fill structures with current keypoints and matches with reference frame
    mvKeys2 = CurrentFrame.mvKeysUn;        // CurrentFrame

    mvMatches12.clear();
    mvMatches12.reserve(mvKeys2.size());    // CurrentFrame

    // mvbMatched1 记录参考帧F1(ReferenceFrame/1)中 每个特征点 是否有 匹配的 特征点
    mvbMatched1.resize(mvKeys1.size());

    // 步骤1：组织特征点对
    /**
     * vMatches12
     * 初始化时 得到的 特征点匹配
     * 大小是 mInitialFrame/ReferenceFrame 1 的特征点数量
     * i:               idx of KeyPoint in ReferenceFrame
     * vMatches12[i]:   = CurrentFrame/2 特征点 序号(idx)
     *                  = -1 (no matching KeyPoint in CurrentFrame/2)
     */
    for(size_t i=0, iend=vMatches12.size();i<iend; i++)
    {
        if(vMatches12[i]>=0)
        {
            /**
             * mvMatches12 储存着匹配点对在参考帧F1(ReferenceFrame)和当前帧F2(CurrentFrame)中的序号
             * typedef pair<int,int> Match;
             * vector<Match> mvMatches12;
             * vector<pair<idx of ReferenceFrame/1, idx of CurrentFrame/2>>
             */
            mvMatches12.push_back(make_pair(i, vMatches12[i]));
            mvbMatched1[i]=true;                        // bool, ReferenceFrame
        }
        else
            mvbMatched1[i]=false;                       // bool, ReferenceFrame
    }
    const int N = mvMatches12.size();                   // 匹配点数

    // Indices for minimum set selection
    vector<size_t> vAllIndices;
    vAllIndices.reserve(N);
    vector<size_t> vAvailableIndices;

    for(int i=0; i<N; i++)
    {
        vAllIndices.push_back(i);
    }


    // =======================================================================================================
    // START 2）特征点对**分组**。这一步主要是为了给 RANSAC 使用，对特征点对按照RANSAC循环次数(mMaxIterations)**随机分组**
    //
    // Generate sets of 8 points for each RANSAC iteration
    //
    // 步骤2：在所有匹配特征点对中随机选择 8对 匹配特征点为一组，共选择 mMaxIterations 组
    // 用于 FindHomography(H Matrix) 和 FindFundamental(F Matrix) 求解
    // mMaxIterations:200
    mvSets = vector< vector<size_t> >(mMaxIterations,vector<size_t>(8,0));

    DUtils::Random::SeedRandOnce(0);

    for(int it=0; it<mMaxIterations; it++)
    {
        vAvailableIndices = vAllIndices;

        // Select a minimum set
        for(size_t j=0; j<8; j++)
        {
            int randi = DUtils::Random::RandomInt(0,vAvailableIndices.size()-1);
            int idx = vAvailableIndices[randi];

            mvSets[it][j] = idx;

            vAvailableIndices[randi] = vAvailableIndices.back();
            vAvailableIndices.pop_back();
        }
    }
    // END 2）特征点对**分组**。这一步主要是为了给 RANSAC 使用，对特征点对按照RANSAC循环次数(mMaxIterations)**随机分组**
    // =====================================================================================================


    // Launch threads to compute in parallel a fundamental matrix and a homography
    // !!! FindHomography -> ComputeH21 -> CheckHomography -> ReconstructH -> get R&t
    // 步骤3：调用多线程分别用于计算 fundamental matrix 和 homography matrix
    vector<bool> vbMatchesInliersH, vbMatchesInliersF;

    // SH 计算 单应矩阵 得分
    // SF 计算 基础矩阵 得分
    float SH, SF;
    cv::Mat H, F;

    // 计算 homograpy 和 得分
    thread threadH(&Initializer::FindHomography, this, ref(vbMatchesInliersH), ref(SH), ref(H));
    // 计算 fundamental 和 得分
    thread threadF(&Initializer::FindFundamental, this, ref(vbMatchesInliersF), ref(SF), ref(F));

    // Wait until both threads have finished
    // 也就是等待SH，SF的结果
    threadH.join();
    threadF.join();

    // Compute ratio of scores
    // 步骤4：计算得分比例，选取某个模型
    float RH = SH/(SH+SF);

    // Try to reconstruct from homography or fundamental depending on the ratio (0.40-0.45)
    // 步骤5：从H矩阵或F矩阵中恢复R,t
    if(RH>0.40)
        return ReconstructH(vbMatchesInliersH,H,mK,R21,t21,vP3D,vbTriangulated,1.0,50);
    else //if(pF_HF>0.6)
        return ReconstructF(vbMatchesInliersF,F,mK,R21,t21,vP3D,vbTriangulated,1.0,50);

    return false;
}


/**
 * 流程
 * 1）特征点归一化
 * 2）计算 单应性矩阵 ComputeH21
 * 3）计算2中矩阵对应的 得分
 * 4）按照设定的RANSAC循环次数循环执行2）和3），并找到 **得分最高** 的那个矩阵
 * 
 * @param vbMatchesInliers  匹配点中哪些可以通过H21重投影成功
 * @param score             输出的得分
 * @param H21               输出的单应性矩阵
 */
void Initializer::FindHomography(vector<bool> &vbMatchesInliers, float &score, cv::Mat &H21)
{
    // Number of putative matches
    // 假定 匹配的数量
    const int N = mvMatches12.size();

    // Normalize coordinates
    vector<cv::Point2f> vPn1, vPn2;
    cv::Mat T1, T2;
    Normalize(mvKeys1,vPn1, T1);
    Normalize(mvKeys2,vPn2, T2);
    cv::Mat T2inv = T2.inv();

    // Best Results variables
    score = 0.0;
    vbMatchesInliers = vector<bool>(N,false);

    // Iteration variables
    vector<cv::Point2f> vPn1i(8);
    vector<cv::Point2f> vPn2i(8);
    cv::Mat H21i, H12i;
    vector<bool> vbCurrentInliers(N,false);
    float currentScore;

    // Perform all RANSAC iterations and save the solution with highest score
    // 在所有 RANSAC 样本中寻找能够使重投影的 点对数达到最多 的样本
    for(int it=0; it<mMaxIterations; it++)
    {
        // Select a minimum set
        for(size_t j=0; j<8; j++)
        {
            int idx = mvSets[it][j];

            vPn1i[j] = vPn1[mvMatches12[idx].first];
            vPn2i[j] = vPn2[mvMatches12[idx].second];
        }
        // 计算本次RANSAC样本下的 单应矩阵
        cv::Mat Hn = ComputeH21(vPn1i,vPn2i);
        H21i = T2inv*Hn*T1;
        H12i = H21i.inv();
        // 在参数 mSigma下，能够通过 H21，H12 重投影成功 的点有哪些 ，并返回分数
        currentScore = CheckHomography(H21i, H12i, vbCurrentInliers, mSigma);

        if(currentScore>score)
        {
            H21 = H21i.clone();
            vbMatchesInliers = vbCurrentInliers;
            score = currentScore;
        }
    }
}


/**
 * @brief
 * 该函数的主要流程如下：
 * 1）特征点归一化
 * 2）计算基础矩阵ComputeF21
 * 3）计算2中矩阵对应的得分
 * 4）按照设定的RANSAC循环次数循环执行2）和3），并找到得分最高的那个矩阵
 * 
 * @param vbMatchesInliers  匹配点中哪些可以通过F21**重投影成功**
 */
void Initializer::FindFundamental(vector<bool> &vbMatchesInliers, float &score, cv::Mat &F21)
{
    // Number of putative matches
    const int N = vbMatchesInliers.size();

    // Normalize coordinates
    vector<cv::Point2f> vPn1, vPn2;
    cv::Mat T1, T2;
    Normalize(mvKeys1, vPn1, T1);
    Normalize(mvKeys2, vPn2, T2);
    cv::Mat T2t = T2.t();

    // Best Results variables
    score = 0.0;
    vbMatchesInliers = vector<bool>(N,false);

    // Iteration variables
    vector<cv::Point2f> vPn1i(8);
    vector<cv::Point2f> vPn2i(8);
    cv::Mat F21i;
    vector<bool> vbCurrentInliers(N,false);
    float currentScore;

    // Perform all RANSAC iterations and save the solution with highest score
    for(int it=0; it<mMaxIterations; it++)
    {
        // Select a minimum set
        for(int j=0; j<8; j++)
        {
            int idx = mvSets[it][j];

            vPn1i[j] = vPn1[mvMatches12[idx].first];
            vPn2i[j] = vPn2[mvMatches12[idx].second];
        }

        // 计算出归一化特征点对应的基础矩阵
        // use 8 pairs of KeyPoints
        cv::Mat Fn = ComputeF21(vPn1i, vPn2i);  // in fact it returns E matrix

        // 转换成归一化前特征点对应的基础矩阵
        F21i = T2t*Fn*T1;                       // this is the real F Matrix

        // 在参数 mSigma下，能够通过F21li，**重投影成功**的点有哪些，并返回分数
        // check every KeyPoint
        currentScore = CheckFundamental(F21i, vbCurrentInliers, mSigma);

        // select the best F21
        if(currentScore>score)
        {
            F21 = F21i.clone();
            vbMatchesInliers = vbCurrentInliers;
            score = currentScore;                   // update the maximum score
        }
    }
}


/**
 * @brief
 * 计算单应性矩阵
 * 
 * @param vP1   帧1中的特征点
 * @param vP2   帧2中的特征点
 */
cv::Mat Initializer::ComputeH21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2)
{
    const int N = vP1.size();

    cv::Mat A(2*N,9,CV_32F);

    for(int i=0; i<N; i++)
    {
        const float u1 = vP1[i].x;
        const float v1 = vP1[i].y;
        const float u2 = vP2[i].x;
        const float v2 = vP2[i].y;

        A.at<float>(2*i,0) = 0.0;
        A.at<float>(2*i,1) = 0.0;
        A.at<float>(2*i,2) = 0.0;
        A.at<float>(2*i,3) = -u1;
        A.at<float>(2*i,4) = -v1;
        A.at<float>(2*i,5) = -1;
        A.at<float>(2*i,6) = v2*u1;
        A.at<float>(2*i,7) = v2*v1;
        A.at<float>(2*i,8) = v2;

        A.at<float>(2*i+1,0) = u1;
        A.at<float>(2*i+1,1) = v1;
        A.at<float>(2*i+1,2) = 1;
        A.at<float>(2*i+1,3) = 0.0;
        A.at<float>(2*i+1,4) = 0.0;
        A.at<float>(2*i+1,5) = 0.0;
        A.at<float>(2*i+1,6) = -u2*u1;
        A.at<float>(2*i+1,7) = -u2*v1;
        A.at<float>(2*i+1,8) = -u2;
    }

    cv::Mat u,w,vt;

    cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    return vt.row(8).reshape(0, 3);
}


/**
 * @brief
 * 通过vP1，vP2求得Essential Matrix并返回
 * 
 * @param vP1   帧1中的特征点
 * @param vP2   帧2中的特征点
 */
cv::Mat Initializer::ComputeF21(const vector<cv::Point2f> &vP1,const vector<cv::Point2f> &vP2)
{
    const int N = vP1.size();

    cv::Mat A(N,9,CV_32F);

    for(int i=0; i<N; i++)
    {
        const float u1 = vP1[i].x;
        const float v1 = vP1[i].y;
        const float u2 = vP2[i].x;
        const float v2 = vP2[i].y;

        A.at<float>(i,0) = u2*u1;
        A.at<float>(i,1) = u2*v1;
        A.at<float>(i,2) = u2;
        A.at<float>(i,3) = v2*u1;
        A.at<float>(i,4) = v2*v1;
        A.at<float>(i,5) = v2;
        A.at<float>(i,6) = u1;
        A.at<float>(i,7) = v1;
        A.at<float>(i,8) = 1;
    }

    cv::Mat u,w,vt;

    cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    cv::Mat Fpre = vt.row(8).reshape(0, 3);

    cv::SVDecomp(Fpre,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    w.at<float>(2)=0;

    return  u*cv::Mat::diag(w)*vt;  // E = uwv
}


/**
 * @brief
 * 评估单应性矩阵
 * 
 * @param H21               单应性矩阵
 * @param H12               单应性矩阵的逆
 * @param vbMatchesInliers  匹配点重投影是否成功的标志位
 * @param sigma             计算得分时需要的参数
 */
float Initializer::CheckHomography(const cv::Mat &H21, const cv::Mat &H12, vector<bool> &vbMatchesInliers, float sigma)
{   
    const int N = mvMatches12.size();

    const float h11 = H21.at<float>(0,0);
    const float h12 = H21.at<float>(0,1);
    const float h13 = H21.at<float>(0,2);
    const float h21 = H21.at<float>(1,0);
    const float h22 = H21.at<float>(1,1);
    const float h23 = H21.at<float>(1,2);
    const float h31 = H21.at<float>(2,0);
    const float h32 = H21.at<float>(2,1);
    const float h33 = H21.at<float>(2,2);

    const float h11inv = H12.at<float>(0,0);
    const float h12inv = H12.at<float>(0,1);
    const float h13inv = H12.at<float>(0,2);
    const float h21inv = H12.at<float>(1,0);
    const float h22inv = H12.at<float>(1,1);
    const float h23inv = H12.at<float>(1,2);
    const float h31inv = H12.at<float>(2,0);
    const float h32inv = H12.at<float>(2,1);
    const float h33inv = H12.at<float>(2,2);

    vbMatchesInliers.resize(N);

    float score = 0;

    // 判断通过单应矩阵重投影是否成功的 阈值
    const float th = 5.991;

    const float invSigmaSquare = 1.0/(sigma*sigma);
    // 遍历所有N对 特征匹配点
    for(int i=0; i<N; i++)
    {
        bool bIn = true;

        const cv::KeyPoint &kp1 = mvKeys1[mvMatches12[i].first];
        const cv::KeyPoint &kp2 = mvKeys2[mvMatches12[i].second];

        const float u1 = kp1.pt.x;
        const float v1 = kp1.pt.y;
        const float u2 = kp2.pt.x;
        const float v2 = kp2.pt.y;

        // Reprojection error in first image
        // x2in1 = H12*x2
        // 将 图像2 中的特征点 单应到 图像1 中
        const float w2in1inv = 1.0/(h31inv*u2+h32inv*v2+h33inv);
        const float u2in1 = (h11inv*u2+h12inv*v2+h13inv)*w2in1inv;
        const float v2in1 = (h21inv*u2+h22inv*v2+h23inv)*w2in1inv;

        // 计算u2，v2投影到F1后与 u1,v1 的距离的平方，也就是重投影误差
        const float squareDist1 = (u1-u2in1)*(u1-u2in1)+(v1-v2in1)*(v1-v2in1);
        // 根据 方差 归一化 误差
        const float chiSquare1 = squareDist1*invSigmaSquare;

        // chiSquare1>th 说明匹配的点对F1投影到F2，重投影失败
        if(chiSquare1>th)
            bIn = false;
        else
            score += th - chiSquare1;

        // Reprojection error in second image
        // x1in2 = H21*x1
        // 将 图像1 中的特征点 单应到 图像2 中
        const float w1in2inv = 1.0/(h31*u1+h32*v1+h33);
        const float u1in2 = (h11*u1+h12*v1+h13)*w1in2inv;
        const float v1in2 = (h21*u1+h22*v1+h23)*w1in2inv;

        const float squareDist2 = (u2-u1in2)*(u2-u1in2)+(v2-v1in2)*(v2-v1in2);

        const float chiSquare2 = squareDist2*invSigmaSquare;

        if(chiSquare2>th)
            bIn = false;
        else
            score += th - chiSquare2;

        // bIn 标志着此对匹配点是否重投影成功
        if(bIn)
            vbMatchesInliers[i]=true;
        else
            vbMatchesInliers[i]=false;
    }

    return score;
}


/**
 * @brief
 * 计算基础得分，判断哪些匹配点**重投影成功**
 * 重投影成功: can find a matched point after reprojection
 * 
 * @param F21               基础矩阵
 * @param vbMatchesInliers  匹配点是否**重投影成功**的标志位
 * @param sigma             计算得分时需要的参数
 */
float Initializer::CheckFundamental(const cv::Mat &F21, vector<bool> &vbMatchesInliers, float sigma)
{
    const int N = mvMatches12.size();

    const float f11 = F21.at<float>(0,0);
    const float f12 = F21.at<float>(0,1);
    const float f13 = F21.at<float>(0,2);
    const float f21 = F21.at<float>(1,0);
    const float f22 = F21.at<float>(1,1);
    const float f23 = F21.at<float>(1,2);
    const float f31 = F21.at<float>(2,0);
    const float f32 = F21.at<float>(2,1);
    const float f33 = F21.at<float>(2,2);

    vbMatchesInliers.resize(N);

    float score = 0;

    const float th = 3.841;
    const float thScore = 5.991;

    const float invSigmaSquare = 1.0/(sigma*sigma);

    for(int i=0; i<N; i++)
    {
        bool bIn = true;

        const cv::KeyPoint &kp1 = mvKeys1[mvMatches12[i].first];
        const cv::KeyPoint &kp2 = mvKeys2[mvMatches12[i].second];

        const float u1 = kp1.pt.x;
        const float v1 = kp1.pt.y;
        const float u2 = kp2.pt.x;
        const float v2 = kp2.pt.y;

        // check each other, so implement two times
        //
        // Reprojection error in second image
        // l2 = F21x1 = (a2,b2,c2)
        // F21*x1 可以算出 x1 在图像中 x2 对应的线 l2 (Epilolar line 2)
        const float a2 = f11*u1+f12*v1+f13;
        const float b2 = f21*u1+f22*v1+f23;
        const float c2 = f31*u1+f32*v1+f33;
        // x2 应该在 l2这条线上: x2点乘l2 = 0
        const float num2 = a2*u2+b2*v2+c2;
        const float squareDist1 = num2*num2/(a2*a2+b2*b2);
        const float chiSquare1 = squareDist1*invSigmaSquare;

        if(chiSquare1>th)
            bIn = false;
        else
            score += thScore - chiSquare1;

        // Reprojection error in first image
        // l1 = x2tF21 = (a1,b1,c1) (Epilolar line 1)
        const float a1 = f11*u2+f21*v2+f31;
        const float b1 = f12*u2+f22*v2+f32;
        const float c1 = f13*u2+f23*v2+f33;
        const float num1 = a1*u1+b1*v1+c1;
        const float squareDist2 = num1*num1/(a1*a1+b1*b1);
        const float chiSquare2 = squareDist2*invSigmaSquare;

        if(chiSquare2>th)
            bIn = false;
        else
            score += thScore - chiSquare2;

        if(bIn)
            vbMatchesInliers[i]=true;
        else
            vbMatchesInliers[i]=false;
    }

    return score;
}


/**
 * @brief
 * 通过输入的F21计算Rt
 * 
 * @param vbMatchesInliers  匹配点中哪些可以通过H21重投影成功
 * @param F21               基础矩阵
 * @param K                 内参
 * @param R21               旋转矩阵
 * @param t21               平移向量
 * @param vP3D              三角化**重投影成功**的匹配点的3d点在 相机1 下的坐标
 * @param vbTriangulated    特征点是否**重投影成功**的标志位
 * @param minParallax       设置的 最小视差角 余弦值参数，输出Rt模型的视差角小于此值则返回失败
 * @param minTriangulated   匹配点中 H21重投影成功的个数 如果小于此值，返回失败
 */
bool Initializer::ReconstructF(vector<bool> &vbMatchesInliers, cv::Mat &F21, cv::Mat &K,
                            cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, 
                            vector<bool> &vbTriangulated, float minParallax, int minTriangulated)
{
    int N=0;
    for(size_t i=0, iend = vbMatchesInliers.size() ; i<iend; i++)
        if(vbMatchesInliers[i])
            N++;

    // Compute Essential Matrix from Fundamental Matrix
    cv::Mat E21 = K.t()*F21*K;

    cv::Mat R1, R2, t;

    // Recover the 4 motion hypotheses
    // 分解本质矩阵
    DecomposeE(E21,R1,R2,t);

    cv::Mat t1=t;
    cv::Mat t2=-t;

    // Reconstruct with the 4 hyphoteses and check
    vector<cv::Point3f> vP3D1, vP3D2, vP3D3, vP3D4;
    vector<bool> vbTriangulated1, vbTriangulated2, vbTriangulated3, vbTriangulated4;
    float parallax1,parallax2, parallax3, parallax4;

    // main part to get R&t from F
    int nGood1 = CheckRT(R1, t1, mvKeys1, mvKeys2, mvMatches12, vbMatchesInliers, K, vP3D1, 4.0*mSigma2, vbTriangulated1, parallax1);
    int nGood2 = CheckRT(R2, t1, mvKeys1, mvKeys2, mvMatches12, vbMatchesInliers, K, vP3D2, 4.0*mSigma2, vbTriangulated2, parallax2);
    int nGood3 = CheckRT(R1, t2, mvKeys1, mvKeys2, mvMatches12, vbMatchesInliers, K, vP3D3, 4.0*mSigma2, vbTriangulated3, parallax3);
    int nGood4 = CheckRT(R2, t2, mvKeys1, mvKeys2, mvMatches12, vbMatchesInliers, K, vP3D4, 4.0*mSigma2, vbTriangulated4, parallax4);

    int maxGood = max(nGood1,max(nGood2,max(nGood3,nGood4)));

    R21 = cv::Mat();
    t21 = cv::Mat();

    // minTriangulated 为可以三角化恢复三维点的个数
    int nMinGood = max(static_cast<int>(0.9*N), minTriangulated);

    int nsimilar = 0;
    if(nGood1>0.7*maxGood)
        nsimilar++;
    if(nGood2>0.7*maxGood)
        nsimilar++;
    if(nGood3>0.7*maxGood)
        nsimilar++;
    if(nGood4>0.7*maxGood)
        nsimilar++;

    // If there is not a clear winner or not enough triangulated points reject initialization
    // nsimilar>1表明没有哪个模型明显胜出
    // 匹配点三角化**重投影成功**数过少
    // 四个结果中如果没有明显的最优结果，则返回失败
    if(maxGood<nMinGood || nsimilar>1)
    {
        return false;
    }

    // If best reconstruction has enough parallax initialize
    if(maxGood==nGood1)
    {
        // 如果模型一对应的视差角大于最小值
        if(parallax1>minParallax)
        {
            vP3D = vP3D1;
            vbTriangulated = vbTriangulated1;

            R1.copyTo(R21);
            t1.copyTo(t21);
            return true;
        }
    }else if(maxGood==nGood2)
    {
        if(parallax2>minParallax)
        {
            vP3D = vP3D2;
            vbTriangulated = vbTriangulated2;

            R2.copyTo(R21);
            t1.copyTo(t21);
            return true;
        }
    }else if(maxGood==nGood3)
    {
        if(parallax3>minParallax)
        {
            vP3D = vP3D3;
            vbTriangulated = vbTriangulated3;

            R1.copyTo(R21);
            t2.copyTo(t21);
            return true;
        }
    }else if(maxGood==nGood4)
    {
        if(parallax4>minParallax)
        {
            vP3D = vP3D4;
            vbTriangulated = vbTriangulated4;

            R2.copyTo(R21);
            t2.copyTo(t21);
            return true;
        }
    }

    return false;
}


/**
 * @brief
 * 其实就是SFM方法，由于恢复的位姿不唯一，需要根据恢复的特征点位姿来判断哪个是正确的
 * 通过输入的H21计算Rt
 * 1: ReferenceFrame
 * 2: CurrentFrame
 * R21: Rotation of CurrentFrame based on ReferenceFrame
 * 
 * @param vbMatchesInliers  匹配点中哪些可以通过H21**重投影成功**
 * @param H21               单应性矩阵
 * @param K                 内参
 * @param R21               旋转矩阵
 * @param t21               平移向量
 * @param vP3D              三角化**重投影成功**的匹配点的3d点在 ReferenceFrame/相机1 下的坐标  -> return
 * @param vbTriangulated    特征点是否**重投影成功**的标志位    -> return
 * @param minParallax       设置的最小视差角余弦值参数，输出Rt模型的视差角小于此值则返回失败
 * @param minTriangulated   匹配点中H21重投影成功的个数如果小于此值，返回失败
 */
bool Initializer::ReconstructH(vector<bool> &vbMatchesInliers, 
                               cv::Mat &H21, cv::Mat &K,
                               cv::Mat &R21, cv::Mat &t21, 
                               vector<cv::Point3f> &vP3D, 
                               vector<bool> &vbTriangulated, 
                               float minParallax, int minTriangulated) 
{
    int N=0;
    for(size_t i=0, iend = vbMatchesInliers.size() ; i<iend; i++)
        if(vbMatchesInliers[i])
            N++;

    // We recover 8 motion hypotheses using the method of Faugeras et al.
    // Motion and structure from motion in a piecewise planar environment.
    // International Journal of Pattern Recognition and Artificial Intelligence, 1988

    // 将H矩阵由 图像坐标系 变换到 相机坐标系
    cv::Mat invK = K.inv();
    cv::Mat A = invK*H21*K;

    cv::Mat U,w,Vt,V;
    cv::SVD::compute(A,w,U,Vt,cv::SVD::FULL_UV);
    V=Vt.t();   //vt转置
    
    float s = cv::determinant(U)*cv::determinant(Vt);   // cv::determinant(U) 为U的行列式

    float d1 = w.at<float>(0);
    float d2 = w.at<float>(1);
    float d3 = w.at<float>(2);

    // 注意d1>d2>d3
    // 看吴博讲解的ppt19页，只考虑d1!=d2!=d3的情况，其他情况返回失败
    if(d1/d2<1.00001 || d2/d3<1.00001)
    {
        return false;
    }

    vector<cv::Mat> vR, vt, vn;
    vR.reserve(8);
    vt.reserve(8);
    vn.reserve(8);

    //n'=[x1 0 x3] 4 posibilities e1=e3=1, e1=1 e3=-1, e1=-1 e3=1, e1=e3=-1
    float aux1 = sqrt((d1*d1-d2*d2)/(d1*d1-d3*d3));
    float aux3 = sqrt((d2*d2-d3*d3)/(d1*d1-d3*d3));
    float x1[] = {aux1,aux1,-aux1,-aux1};
    float x3[] = {aux3,-aux3,aux3,-aux3};

    //case d'=d2
    float aux_stheta = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1+d3)*d2);

    float ctheta = (d2*d2+d1*d3)/((d1+d3)*d2);
    float stheta[] = {aux_stheta, -aux_stheta, -aux_stheta, aux_stheta};

    for(int i=0; i<4; i++)
    {
        cv::Mat Rp=cv::Mat::eye(3,3,CV_32F);
        Rp.at<float>(0,0)=ctheta;
        Rp.at<float>(0,2)=-stheta[i];
        Rp.at<float>(2,0)=stheta[i];
        Rp.at<float>(2,2)=ctheta;

        cv::Mat R = s*U*Rp*Vt;
        vR.push_back(R);

        cv::Mat tp(3,1,CV_32F);
        tp.at<float>(0)=x1[i];
        tp.at<float>(1)=0;
        tp.at<float>(2)=-x3[i];
        tp*=d1-d3;

        cv::Mat t = U*tp;
        vt.push_back(t/cv::norm(t));

        cv::Mat np(3,1,CV_32F);
        np.at<float>(0)=x1[i];
        np.at<float>(1)=0;
        np.at<float>(2)=x3[i];

        cv::Mat n = V*np;
        if(n.at<float>(2)<0)
            n=-n;
        vn.push_back(n);
    }

    //case d'=-d2
    float aux_sphi = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1-d3)*d2);

    float cphi = (d1*d3-d2*d2)/((d1-d3)*d2);
    float sphi[] = {aux_sphi, -aux_sphi, -aux_sphi, aux_sphi};

    for(int i=0; i<4; i++)
    {
        cv::Mat Rp=cv::Mat::eye(3,3,CV_32F);
        Rp.at<float>(0,0)=cphi;
        Rp.at<float>(0,2)=sphi[i];
        Rp.at<float>(1,1)=-1;
        Rp.at<float>(2,0)=sphi[i];
        Rp.at<float>(2,2)=-cphi;

        cv::Mat R = s*U*Rp*Vt;
        vR.push_back(R);

        cv::Mat tp(3,1,CV_32F);
        tp.at<float>(0)=x1[i];
        tp.at<float>(1)=0;
        tp.at<float>(2)=x3[i];
        tp*=d1+d3;

        cv::Mat t = U*tp;
        vt.push_back(t/cv::norm(t));

        cv::Mat np(3,1,CV_32F);
        np.at<float>(0)=x1[i];
        np.at<float>(1)=0;
        np.at<float>(2)=x3[i];

        cv::Mat n = V*np;
        if(n.at<float>(2)<0)
            n=-n;
        vn.push_back(n);
    }


    int bestGood = 0;
    int secondBestGood = 0;    
    int bestSolutionIdx = -1;
    float bestParallax = -1;
    vector<cv::Point3f> bestP3D;
    vector<bool> bestTriangulated;

    // Instead of applying the visibility constraints proposed in the Faugeras' paper 
    // (which could fail for points seen with low parallax)
    // We reconstruct all hypotheses and check in terms of 
    // triangulated points and parallax
    // 经过上面的计算，共有8种R、t计算结果，遍历这8种可能模型
    // 通过计算出匹配点的三角化**重投影成功**的数量，来找出**最好模型**和**次好模型**
    for(size_t i=0; i<8; i++)
    {
        float parallaxi;
        vector<cv::Point3f> vP3Di;
        vector<bool> vbTriangulatedi;

        // 计算在输入Rt下，匹配点三角化**重投影成功**的数量
        // return: vP3Di, vbTriangulatedi, parallaxi
        // check the return later
        // main part to get R&t from H
        int nGood = CheckRT(vR[i], vt[i], mvKeys1, mvKeys2, mvMatches12, vbMatchesInliers, K, vP3Di, 4.0*mSigma2, vbTriangulatedi, parallaxi);

        if(nGood>bestGood)                      // update the best and second good results
        {
            secondBestGood = bestGood;
            bestGood = nGood;
            bestSolutionIdx = i;
            bestParallax = parallaxi;
            bestP3D = vP3Di;
            bestTriangulated = vbTriangulatedi;
        }
        else if(nGood>secondBestGood)           // update the second good results
        {
            secondBestGood = nGood;
        }
    }

    /**
     * secondBestGood<0.75*bestGood 如果最好模型与次好模型 差距 足够大
     * bestParallax>=minParallax    最好模型对应的 视差角 大于此值
     * bestGood>minTriangulated     最好模型对应的 匹配点三角化重投影成功 数量 大于此阈值
     * bestGood>0.9*N               匹配点三角化**重投影成功**数量 占 通过H**重投影成功**数量 的比例需要 大于0.9
     */
    if(secondBestGood<0.75*bestGood && bestParallax>=minParallax && bestGood>minTriangulated && bestGood>0.9*N)
    {
        vR[bestSolutionIdx].copyTo(R21);
        vt[bestSolutionIdx].copyTo(t21);
        vP3D = bestP3D;                         // return
        vbTriangulated = bestTriangulated;      // return

        return true;
    }

    return false;
}


/**
 * @brief
 * 计算kp1,kp2是匹配的关键点，它对应着世界坐标系中的一个点。
 * P1,P2是F1，F2对应的投影矩阵。
 * 输出综合考虑了P1,P2,kp1,kp2的在世界坐标系中的齐次坐标3D点坐标
 * 
 * @param kp1 
 * @param kp2 
 * @param P1 
 * @param P2
 * @param x3D   输出综合考虑了P1,P2,kp1,kp2的在世界坐标系中的齐次坐标3D点坐标
 */
void Initializer::Triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D)
{
    cv::Mat A(4,4,CV_32F);

    // A * x = 0
    A.row(0) = kp1.pt.x*P1.row(2)-P1.row(0);    // u*P3 - P1
    A.row(1) = kp1.pt.y*P1.row(2)-P1.row(1);    // v*P3 - P2
    A.row(2) = kp2.pt.x*P2.row(2)-P2.row(0);
    A.row(3) = kp2.pt.y*P2.row(2)-P2.row(1);

    /**
     * Performs SVD of a matrix
     * C++: static void SVD::compute(InputArray src, OutputArray w, OutputArray u, OutputArray vt, int flags=0 )
     * C++: static void SVD::compute(InputArray src, OutputArray w, int flags=0 )
     * 
     * Parameters:
     * src – decomposed matrix
     * w – calculated singular values (D)
     * u – calculated left singular vectors
     * vt – transposed matrix of right singular values (v)
     * flags – operation flags
     *  SVD::MODIFY_A   use the algorithm to modify the decomposed matrix; 
     *                  it can save space and speed up processing.
     *  SVD::NO_UV      indicates that only a vector of singular values w is to be processed, 
     *                  while u and vt will be set to empty matrices.
     *  SVD::FULL_UV    when the matrix is not square, by default the algorithm produces u and vt matrices 
     *                  of sufficiently large size for the further A reconstruction; 
     *                  if, however, FULL_UV flag is specified, u and vt will be full-size square orthogonal matrices.
     */
    cv::Mat u,w,vt; // vt: 4d
    cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
    
    x3D = vt.row(3).t();                        // 取最后一个行向量
    x3D = x3D.rowRange(0,3)/x3D.at<float>(3);   // 转换为齐次坐标, [x, y, z, 1] -> [x, y, z]
}


/**
 * @brief
 * 将一个特征点集合归一化到另一个坐标系，使得归一化后的坐标点集合均值为0，一阶绝对矩为1，这样计算矩阵更准确
 * 
 * @param vKeys                 待归一化 特征点集合
 * @param vNormalizedPoints     归一化后 特征点集合 -> return
 * @param T                     归一化 所使用的矩阵 -> return
 */
void Initializer::Normalize(const vector<cv::KeyPoint> &vKeys, 
    vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T) {
    float meanX = 0;
    float meanY = 0;
    const int N = vKeys.size();

    vNormalizedPoints.resize(N);

    for(int i=0; i<N; i++)
    {
        meanX += vKeys[i].pt.x;
        meanY += vKeys[i].pt.y;
    }

    meanX = meanX/N;
    meanY = meanY/N;

    float meanDevX = 0;
    float meanDevY = 0;

    // 将所有 vKeys 点**减去中心坐标**，使x坐标和y坐标均值分别为0
    for(int i=0; i<N; i++)
    {
        vNormalizedPoints[i].x = vKeys[i].pt.x - meanX;
        vNormalizedPoints[i].y = vKeys[i].pt.y - meanY;

        meanDevX += fabs(vNormalizedPoints[i].x);
        meanDevY += fabs(vNormalizedPoints[i].y);
    }

    meanDevX = meanDevX/N;
    meanDevY = meanDevY/N;

    float sX = 1.0/meanDevX;
    float sY = 1.0/meanDevY;

    // 将x坐标和y坐标分别进行**缩放**，使得x坐标和y坐标的一阶绝对矩分别为1
    for(int i=0; i<N; i++)
    {
        vNormalizedPoints[i].x = vNormalizedPoints[i].x * sX;
        vNormalizedPoints[i].y = vNormalizedPoints[i].y * sY;
    }

    /**
     * T    [sX   0  -meanX*sX]
     *      [ 0  sY  -meanY*sY]
     *      [ 0   0          1]
     */
    T = cv::Mat::eye(3,3,CV_32F);
    T.at<float>(0,0) = sX;
    T.at<float>(1,1) = sY;
    T.at<float>(0,2) = -meanX*sX;
    T.at<float>(1,2) = -meanY*sY;
}


/**
 * @brief
 * 计算在输入Rt下，匹配点三角化重投影成功的数量
 * 
 * @param R                 旋转矩阵
 * @param t                 平移矩阵
 * @param vKeys1            帧1 的 特征点
 * @param vKeys2            帧2 的 特征点
 * @param vMatches12        orbmatcher 计算的**初匹配**
 * @param vbMatchesInliers  匹配点中 哪些可以 通过 H 或者 F **重投影成功**
 * @param K                 相机内参
 * @param vP3D              三角化**重投影成功**的匹配点的3d点 在相机1下的坐标 -> return
 * @param th2               根据 三角化重投影 误差 判断 匹配点 是否**重投影成功**的阈值
 * @param vbGood            特征点哪些三角化**重投影成功** -> return
 * @param parallax          三角化**重投影成功**匹配点的**视差角** -> return
 */
int Initializer::CheckRT(const cv::Mat &R, const cv::Mat &t, 
                         const vector<cv::KeyPoint> &vKeys1, 
                         const vector<cv::KeyPoint> &vKeys2,
                         const vector<Match> &vMatches12, 
                         vector<bool> &vbMatchesInliers,
                         const cv::Mat &K, 
                         vector<cv::Point3f> &vP3D, 
                         float th2, 
                         vector<bool> &vbGood, 
                         float &parallax)
{
    // Calibration parameters
    const float fx = K.at<float>(0,0);
    const float fy = K.at<float>(1,1);
    const float cx = K.at<float>(0,2);
    const float cy = K.at<float>(1,2);

    vbGood = vector<bool>(vKeys1.size(),false); // return
    vP3D.resize(vKeys1.size());                 // return

    vector<float> vCosParallax;
    vCosParallax.reserve(vKeys1.size());

    // Camera 1 Projection Matrix K[I|0]
    /** 
     * Operation: Extracts a rectangular submatrix
     * C++: Mat Mat::operator()(Range rowRange, Range colRange) const
     * C++: Mat Mat::operator()(const Rect& roi) const
     * C++: Mat Mat::operator()(const Range* ranges) const
     * 
     * Parameters:
     * rowRange – Start and end row of the extracted submatrix. The upper boundary is not included. To select all the rows, use Range::all().
     * colRange – Start and end column of the extracted submatrix. The upper boundary is not included. To select all the columns, use Range::all().
     * roi – Extracted submatrix specified as a rectangle.
     * ranges – Array of selected ranges along each array dimension.
     * 
     * Example:
     * Mat Test = (Mat_<double>(3,3) << 0,1,2, 3,4,5, 6,7,8);  
     * cout << "Total matrix:" << endl;  
     * cout << Test << endl;
     * Mat testrow = Test.rowRange(0,1).clone();  
     * cout << testrow << endl;  
     * cout << Test.row(0) << endl;
     * 
     * show same result [0, 1, 2]
     * 
     * ===============================================
     * 相机1 的 投影矩阵 K[I|0]，世界坐标系和相机1坐标系相同
     * 
     * P1 [fx  0 cx]
     *    [0  fy cy]
     *    [0   0  1]
     */
    cv::Mat P1(3, 4, CV_32F, cv::Scalar(0));    // K
    K.copyTo(P1.rowRange(0,3).colRange(0,3));   // Copies K to P1
    
    // 相机1 的 光心 在世界坐标系坐标
    cv::Mat O1 = cv::Mat::zeros(3,1,CV_32F);

    // Camera 2 Projection Matrix K[R|t]
    /**
     * 相机2 的 投影矩阵
     * P2 [r11 r12 r13 t1]
     *    [r21 r22 r23 t2]
     *    [r31 r32 r33 t3]
     */
    cv::Mat P2(3,4,CV_32F);
    R.copyTo(P2.rowRange(0,3).colRange(0,3));   // Copies R to P2
    t.copyTo(P2.rowRange(0,3).col(3));          // Copies t to P2
    P2 = K*P2;

    // 相机2的光心在 世界坐标系 坐标
    cv::Mat O2 = -R.t()*t;

    int nGood=0;

    // 遍历所有的 匹配点
    for(size_t i=0, iend=vMatches12.size();i<iend;i++)
    {
        if(!vbMatchesInliers[i])
            continue;                                       // select

        // kp1和kp2是匹配特征点
        const cv::KeyPoint &kp1 = vKeys1[vMatches12[i].first];  // vMatches12[i].first:  idx of KeyPoint in KF1
        const cv::KeyPoint &kp2 = vKeys2[vMatches12[i].second]; // vMatches12[i].second: idx of KeyPoint in KF2

        // 3d点在 相机1/世界坐标系/ReferenceFrame 下的坐标
        cv::Mat p3dC1; // point 3d in C1/world coordinate -> return

        // 输出的 p3dC1 是综合考虑了 P1 , P2 的kp1,kp2 匹配点 在世界坐标系中的 齐次坐标
	    // 由于世界坐标系和相机1坐标系 重合，所以 p3dC1 同时也是 匹配点 对应的空间点 在 相机1 坐标系中的 坐标
        Triangulate(kp1, kp2, P1, P2, p3dC1);                   // return p3dC1, compute MapPoint position

        // isfinite()判断一个浮点数是否是一个有限值
	    // 相当于是确定 p3dC1 前三位数值正常
        if(!isfinite(p3dC1.at<float>(0)) || !isfinite(p3dC1.at<float>(1)) || !isfinite(p3dC1.at<float>(2)))
        {
            vbGood[vMatches12[i].first]=false;
            continue;                                       // select
        }


        // Check parallax
        // normal1 是相机1 到 3d点 的向量
        cv::Mat normal1 = p3dC1 - O1;
        float dist1 = cv::norm(normal1);

        // normal2 是相机2 到 3d点 的向量
        cv::Mat normal2 = p3dC1 - O2;
        float dist2 = cv::norm(normal2);

        // cosParallax 为视差角的余弦，也就是 normal1 与 normal2 的余弦
        float cosParallax = normal1.dot(normal2)/(dist1*dist2);

        // Check depth in front of first camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        // 判断3D点是否在两个摄像头前方
        // p3dC1.at<float>(2)<=0 说明3d点在 光心 后面，深度为负
	    // p3dC1 视差角 较大，且深度为负 则淘汰
        if(p3dC1.at<float>(2)<=0 && cosParallax<0.99998)
            continue;                                       // select

        // Check depth in front of second camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        cv::Mat p3dC2 = R*p3dC1+t;
        // p3dC2 视差角 较大，且深度为负 则淘汰
        if(p3dC2.at<float>(2)<=0 && cosParallax<0.99998)
            continue;                                       // select


        // Check reprojection error in first image
        // 计算3D点在第一个图像上的投影误差
        float im1x, im1y;
        float invZ1 = 1.0/p3dC1.at<float>(2);
        im1x = fx*p3dC1.at<float>(0)*invZ1+cx;
        im1y = fy*p3dC1.at<float>(1)*invZ1+cy;

        float squareError1 = (im1x-kp1.pt.x)*(im1x-kp1.pt.x)+(im1y-kp1.pt.y)*(im1y-kp1.pt.y);   // estimate value - observation
        // 重投影误差太大，淘汰
        if(squareError1>th2)
            continue;                                       // select


        // Check reprojection error in second image
        float im2x, im2y;
        float invZ2 = 1.0/p3dC2.at<float>(2);
        im2x = fx*p3dC2.at<float>(0)*invZ2+cx;
        im2y = fy*p3dC2.at<float>(1)*invZ2+cy;

        float squareError2 = (im2x-kp2.pt.x)*(im2x-kp2.pt.x)+(im2y-kp2.pt.y)*(im2y-kp2.pt.y);

        if(squareError2>th2)
            continue;                                       // select


        // 统计经过检验的 3D点 个数，记录3D点视差角
        // 到这里说明这对匹配点三角化**重投影成功**了
        vCosParallax.push_back(cosParallax);
        vP3D[vMatches12[i].first] = cv::Point3f(p3dC1.at<float>(0),p3dC1.at<float>(1),p3dC1.at<float>(2));  // vMatches12[i].first:  idx of KeyPoint in KF1
        nGood++;
        // 确认视差角够大
        if(cosParallax<0.99998)
            vbGood[vMatches12[i].first]=true;               // criterion for good point
    }

    // 得到3D点中较大的 视差角
    if(nGood>0)
    {
        // 将视差角余弦有小到大**排序**
        sort(vCosParallax.begin(),vCosParallax.end());
        // 取出第50个，或者最后那个也就是最大那个
        size_t idx = min(50,int(vCosParallax.size()-1));
        // 计算出视差角
        parallax = acos(vCosParallax[idx])*180/CV_PI;
    }
    else
        parallax=0;

    return nGood;
}


/**
 * @brief
 * 将本质矩阵分解成Rt,有四种模型
 * 
 * @param E 
 * @param R1    输出其中一种R
 * @param R2    输出其中一种R
 * @param t     输出t
 */
void Initializer::DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t)
{
    cv::Mat u,w,vt;
    cv::SVD::compute(E,w,u,vt);

    u.col(2).copyTo(t);
    t=t/cv::norm(t);

    /**
     * W [0 -1  0]
     *   [1  0  0]
     *   [0  0  1]
     */
    cv::Mat W(3,3,CV_32F,cv::Scalar(0));
    W.at<float>(0,1)=-1;
    W.at<float>(1,0)=1;
    W.at<float>(2,2)=1;

    R1 = u*W*vt;
    if(cv::determinant(R1)<0)
        R1=-R1;

    R2 = u*W.t()*vt;
    if(cv::determinant(R2)<0)
        R2=-R2;
}

} //namespace ORB_SLAM
