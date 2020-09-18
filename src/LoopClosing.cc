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

#include "LoopClosing.h"

#include "Sim3Solver.h"

#include "Converter.h"

#include "Optimizer.h"

#include "ORBmatcher.h"

#include<mutex>
#include<thread>


namespace ORB_SLAM2
{

LoopClosing::LoopClosing(Map *pMap, KeyFrameDatabase *pDB, ORBVocabulary *pVoc, const bool bFixScale):
    mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(pMap),
    mpKeyFrameDB(pDB), mpORBVocabulary(pVoc), mpMatchedKF(NULL), mLastLoopKFid(0), mbRunningGBA(false), mbFinishedGBA(true),
    mbStopGBA(false), mpThreadGBA(NULL), mbFixScale(bFixScale), mnFullBAIdx(0)
{
    mnCovisibilityConsistencyTh = 3;
}

void LoopClosing::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}

void LoopClosing::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}


void LoopClosing::Run()
{
    mbFinished =false;

    while(1)
    {
        // Check if there are keyframes in the queue
        // 如果有新的keyframe插入到闭环检测序列（在localmapping::run()结尾处插入）
        if(CheckNewKeyFrames())
        {
            // Detect loop candidates and check covisibility consistency
            // 检测是否有闭环候选关键帧
            if(DetectLoop())
            {
                // Compute similarity transformation [sR|t]
                // In the stereo/RGBD case s=1
                // 计算候选关键帧的与当前帧的sim3并且返回是否形成闭环的判断
		        // 并在候选帧中找出闭环帧
		        // 并计算出当前帧和闭环帧的sim3
                if(ComputeSim3())
                {
                    // Perform loop fusion and pose graph optimization
                    CorrectLoop();
                }
            }
        }       

        ResetIfRequested();

        if(CheckFinish())
            break;

        usleep(5000);
    }

    SetFinish();
}

void LoopClosing::InsertKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexLoopQueue);
    if(pKF->mnId!=0)
        mlpLoopKeyFrameQueue.push_back(pKF);
}

bool LoopClosing::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexLoopQueue);
    return(!mlpLoopKeyFrameQueue.empty());
}


bool LoopClosing::DetectLoop()
{
    // 先将要处理的闭环检测队列的关键帧弹出来一个
    {
        unique_lock<mutex> lock(mMutexLoopQueue);
        mpCurrentKF = mlpLoopKeyFrameQueue.front();
        mlpLoopKeyFrameQueue.pop_front();
        // Avoid that a keyframe can be erased while it is being process by this thread
        mpCurrentKF->SetNotErase();
    }

    // If the map contains less than 10 KF or less than 10 KF have passed from last loop detection
    // 如果距离上次闭环没多久（小于10帧），或者map中关键帧总共还没有10帧，则不进行闭环检测
    if(mpCurrentKF->mnId<mLastLoopKFid+10)
    {
        mpKeyFrameDB->add(mpCurrentKF);
        mpCurrentKF->SetErase();
        return false;
    }

    // Compute reference BoW similarity score
    // This is the **lowest score** to a connected keyframe in the covisibility graph
    // We will impose loop candidates to have a **higher similarity** than this

    //返回Covisibility graph中与此节点连接的节点（即关键帧），总的来说这一步是为了计算阈值 minScore
    const vector<KeyFrame*> vpConnectedKeyFrames = mpCurrentKF->GetVectorCovisibleKeyFrames();
    const DBoW2::BowVector &CurrentBowVec = mpCurrentKF->mBowVec;
    float minScore = 1;
    for(size_t i=0; i<vpConnectedKeyFrames.size(); i++)
    {
        KeyFrame* pKF = vpConnectedKeyFrames[i];
        if(pKF->isBad())
            continue;
        const DBoW2::BowVector &BowVec = pKF->mBowVec;

        //遍历所有共视关键帧，计算当前关键帧与每个共视关键的bow相似度得分，计算 minScore
        float score = mpORBVocabulary->score(CurrentBowVec, BowVec);    // minscore

        if(score<minScore)
            minScore = score;
    }

    // Query the database imposing the minimum score
    // 在最低相似度 minScore 的要求下，获得闭环检测的**候选帧集合**
    // DetectLoopCandidates()忽略和自己已有共视关系的关键帧
    //
    // 在除去当前帧共视关系的关键帧数据中，检测闭环候选帧(这个函数在KeyFrameDatabase中)
    // 闭环候选帧删选过程：
    // 1. BoW得分>minScore;
    // 2. 统计满足1的关键帧中有共同单词最多的单词数maxcommonwords
    // 3. **筛选**出共同单词数大于mincommons(=0.8*maxcommons)的**关键帧**
    // 4. 相连的关键帧分为一组，计算组得分（总分）,得到最大总分bestAccScore,筛选出总分大于minScoreToRetain(=0.75*bestAccScore)的组
    //    用得分最高的候选帧IAccScoreAndMathch代表该组，计算组得分的目的是剔除单独一帧得分较高，但是没有共视关键帧作为闭环来说不够鲁棒
    //    对于通过了闭环检测的关键帧，还需要通过连续性检测(连续三帧都通过上面的筛选)，才能作为闭环候选帧
    
    /**
     * https://blog.csdn.net/liu502617169/article/details/90286854 
     * vpCandidateKFs里的关键帧A, B,…, M, N表示着可能和当前关键帧形成闭环的地点。
     * 我们把这些地点对应的设为A, B,…, M, N，这时候一个地点对应一个关键帧（对应图1中红色有x的圈圈）。
     * 
     * # 如果我们回忆发现M这个地点在已经以前连续见过2次了，那么代表M这个地点的关键帧就满足连续性条件判断放入mvpEnoughConsistentCandidates中；
     *   地点A, B,…, M, N用一个关键帧来描述不大准确，于是对于每个地点本来就有的关键帧我们就把它扩展，
     * # 只要和这个关键帧在Covisibility graph连接的（图1黑色空心圆），都把它们拉进来分别形成了spCandidateGroup
     * # spCandidateGroup中由于黑色圆和红色圆具有共视关系，所以它们相互之间在空间中聚集在一起。因此，spCandidateGroup能**更加全面的描述一个地点**
     * # mvConsistentGroups 是由 ConsistentGroup 组成的vector
     * # 每个 ConsistentGroup 有一个**关键帧集合**和一个数字 consistency
     *   ConsistentGroup 这个关键帧集合以前是由 spCandidateGroup 转化而来的，也就是说它们彼此之间空间距离相近。
     *   这也就意味着每个 ConsistentGroup 描述着一个地点，其数字consistency记录了这个地点以前我们连续见过多少次。
     * # 因此， mvConsistentGroups 就好似一个我们的一个小本子，记录着a,b,…,g,h这些地点我们以前连续见过多少次。
     */
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectLoopCandidates(mpCurrentKF, minScore);   // candidate KeyFrames

    // If there are no loop candidates, just add new keyframe and return false
    if(vpCandidateKFs.empty())
    {
        mpKeyFrameDB->add(mpCurrentKF);
        mvConsistentGroups.clear();
        mpCurrentKF->SetErase();
        return false;
    }

    // For each loop candidate check consistency with previous loop candidates
    // Each candidate expands a covisibility group (keyframes connected to the loop candidate in the covisibility graph)
    // A group is consistent with a previous group if they share at least a keyframe
    // We must detect a consistent loop in several consecutive keyframes to accept it
    // vpCandidateKFs 中的每个闭环检测的候选帧都会通过共视关键帧，扩展为一个 spCandidateGroup
    // 对于这vpCandidateKFs.size()个spCandidateGroup进行连续性（consistency）判断

    mvpEnoughConsistentCandidates.clear();

    vector<ConsistentGroup> vCurrentConsistentGroups;
    vector<bool> vbConsistentGroup(mvConsistentGroups.size(),false);

    // 遍历vpCandidateKFs，将其中每个关键帧都通过寻找在covisibility graph与自己连接的关键帧，扩展为一个 spCandidateGroup
    // 也就是遍历每一个spCandidateGroup
    // FOR1
    // 遍历每个的spCandidateGroup
    for(size_t i=0, iend=vpCandidateKFs.size(); i<iend; i++)    // v: std::vector<>
    {
        KeyFrame* pCandidateKF = vpCandidateKFs[i];

        // 这个条件是否太宽松?pCandidateKF->GetVectorCovisibleKeyFrames()是否更好一点？
        set<KeyFrame*> spCandidateGroup = pCandidateKF->GetConnectedKeyFrames();    // s: std::set<>
        spCandidateGroup.insert(pCandidateKF);  // covisibile graph

        bool bEnoughConsistent = false;
        bool bConsistentForSomeGroup = false;

        // 遍历 mvConsistentGroups ，判断spCandidateGroup与mvConsistentGroups[iG]是否**连续**
        // current frame's covisibile graph vs covisibile graph of frame of current frame's covisibile graph
	    // FOR2 
        // 遍历mvConsistentGroups中的每个的ConsistentGroup为sPreviousGroup
        // 我们需要判断spCandidateGroup和sPreviousGroup是否有相同的关键帧
        for(size_t iG=0, iendG=mvConsistentGroups.size(); iG<iendG; iG++)
        {
            set<KeyFrame*> sPreviousGroup = mvConsistentGroups[iG].first;

            // 当前的spCandidateGroup之后要不要插入 vCurrentConsistentGroups
            bool bConsistent = false;

            // 遍历spCandidateGroup里的关键帧，判断spCandidateGroup与mvConsistentGroups[iG]是否连续，
	        // 也就是判断spCandidateGroup和mvConsistentGroups[iG]是否有**相同的关键帧**
	        // FOR3
            // 遍历spCandidateGroup中的关键帧，看是否能在sPreviousGroup中找到
            /**
             * spCandidateGroup和sPreviousGroup有相同的关键帧意味着什么？
             * 你想，假如spCandidateGroup的关键帧集合描述的是地点M，它们在地点M周围聚集。
             * sPreviousGroup的关键帧集合描述的是地点g，它们在地点g周围聚集。
             * spCandidateGroup和sPreviousGroup有相同的关键帧就意味着M和g地点就描述的是同一地点。
             * 
             * 于是就有以下3种情况：
             * # B与h有相同的关键帧
             *   说明B和h是相同的一个地点，而h的consistency=2，
             *   说明B，h代表的地点我们已经以前连续见过2次了，加上现在这一次总共连续3次了。
             *   **因此**满足连续性条件**，把最能代表B，h这个地点的候选关键帧（也就是B的红色圈圈）
             *   放入了 mvpEnoughConsistentCandidates
             * # B与g有相同的关键帧，
             *   说明B和g是相同的一个地点，而g的consistency=1,
             *   说明B，h代表的地点我们已经以前连续见过1次了，加上现在这一次总共连续2次了。
             *   **以后有可能**满足连续性条件**，待以后考察，于是把它和consistency=2形成一个
             *   新的 ConsistentGroup 放入 vCurrentConsistentGroups 中。
             * # N没有在 mvConsistentGroups 中找到具有相同关键帧的 ConsistentGroup ，
             *   于是把它和consistency=1形成一个新的 ConsistentGroup 放入 vCurrentConsistentGroups 中
             * 
             * 注意，假设 mvConsistentGroups 中代表地点b的ConsistentGroup没有任何的spCandidateGroup有相同的关键帧，
             * 那么在这一轮的考察中，这个地点b将会被舍去。
             */
            for(set<KeyFrame*>::iterator sit=spCandidateGroup.begin(), send=spCandidateGroup.end(); sit!=send;sit++)
            {
                // 如果在sPreviousGroup里找到sit
                if(sPreviousGroup.count(*sit))
                {
                    // true表示标记sit所在的spCandidateGroup与sPreviousGroup连续（consistent）
		            // 之后要插入到 vCurrentConsistentGroups
                    bConsistent=true;
                    // true表示有 spCandidateGroup 与 vCurrentConsistentGroups 中的元素存在连续（consistent）
                    bConsistentForSomeGroup=true;
                    break;
                }
            }

            if(bConsistent)
            {
                int nPreviousConsistency = mvConsistentGroups[iG].second;
                int nCurrentConsistency = nPreviousConsistency + 1;
                if(!vbConsistentGroup[iG])
                {
                    ConsistentGroup cg = make_pair(spCandidateGroup, nCurrentConsistency);
                    vCurrentConsistentGroups.push_back(cg);
                    vbConsistentGroup[iG]=true;     // this avoid to include the same group more than once
                }
                if(nCurrentConsistency>=mnCovisibilityConsistencyTh && !bEnoughConsistent)
                {
                    mvpEnoughConsistentCandidates.push_back(pCandidateKF);
                    bEnoughConsistent=true;         // this avoid to insert the same candidate more than once
                }
            }
        }

        // If the group is not consistent with any previous group insert with consistency counter set to zero
        if(!bConsistentForSomeGroup)
        {
            ConsistentGroup cg = make_pair(spCandidateGroup,0);
            vCurrentConsistentGroups.push_back(cg);
        }
    }

    // Update Covisibility Consistent Groups
    mvConsistentGroups = vCurrentConsistentGroups;

    // Add Current Keyframe to database
    mpKeyFrameDB->add(mpCurrentKF);

    // vpCandidateKFs 中的闭环候选帧通过连续性判断筛选后得到 mvpEnoughConsistentCandidates
    if(mvpEnoughConsistentCandidates.empty())
    {
        mpCurrentKF->SetErase();
        return false;
    }
    else
    {
        return true;
    }

    mpCurrentKF->SetErase();
    return false;
}


/**
 * 1. 候选帧和当前关键帧通过Bow加速描述子的**匹配**，剔除特征点匹配数少的闭环候选帧
 * 2. 利用RANSAC粗略地计算出当前帧与闭环帧的Sim3，**选出较好的那个**sim3**，确定闭环帧
 * 3. 根据确定的闭环帧和对应的Sim3，对3D点进行投影**找到更多匹配**，通过优化的方法计算**更精确的Sim3**。
 * 4. 将闭环帧以及闭环帧相连的关键帧的MapPoints与当前帧的点进行**匹配**（当前帧---闭环帧+相连关键帧）
 */
bool LoopClosing::ComputeSim3()
{
    // For each consistent loop candidate we try to compute a Sim3

    const int nInitialCandidates = mvpEnoughConsistentCandidates.size();

    // We compute first ORB matches for each candidate
    // If enough matches are found, we setup a Sim3Solver
    ORBmatcher matcher(0.75,true);

    vector<Sim3Solver*> vpSim3Solvers;
    vpSim3Solvers.resize(nInitialCandidates);

    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nInitialCandidates);

    // 标识nInitialCandidates中哪些keyframe将被抛弃
    vector<bool> vbDiscarded;
    vbDiscarded.resize(nInitialCandidates);

    // 候选关键帧和当前帧满足有充足匹配条件的个数
    int nCandidates=0; //candidates with enough matches

    // **遍历**候选闭环关键帧 nInitialCandidates
    // 闭环关键帧与关键帧特征**匹配**，通过bow加速
    // **剔除**特征点匹配数少的闭环候选帧
    for(int i=0; i<nInitialCandidates; i++)
    {
        KeyFrame* pKF = mvpEnoughConsistentCandidates[i];   // candidate keyframe

        // avoid that local mapping erase it while it is being processed in this thread
        // 防止在 LocalMapping 中 KeyFrameCulling 函数将此关键帧作为冗余帧剔除
        pKF->SetNotErase();

        if(pKF->isBad())
        {
            // 舍弃该帧
            vbDiscarded[i] = true;
            continue;
        }

        // 步骤2：将当前帧mpCurrentKF与闭环候选关键帧pKF**匹配**
        // 匹配 mpCurrentKF 与 pKF 之间的特征点并通过bow加速
        // vvpMapPointMatches[i][j]表示mpCurrentKF的特征点j通过mvpEnoughConsistentCandidates[i]匹配得到的mappoint
        int nmatches = matcher.SearchByBoW(mpCurrentKF,pKF,vvpMapPointMatches[i]);

        // 匹配的特征点数量太少，剔除该候选关键帧
        if(nmatches<20)
        {
            vbDiscarded[i] = true;
            continue;
        }
        else
        {
            // 新建一个Sim3Solver对象，执行其构造函数
            Sim3Solver* pSolver = new Sim3Solver(mpCurrentKF,pKF,vvpMapPointMatches[i],mbFixScale);
            pSolver->SetRansacParameters(0.99,20,300);
            vpSim3Solvers[i] = pSolver;
        }

        nCandidates++;
    }

    bool bMatch = false;

    // Perform alternatively RANSAC iterations for each candidate
    // until one is succesful or all fail
    while(nCandidates>0 && !bMatch)
    {
        // 遍历nInitialCandidates
        for(int i=0; i<nInitialCandidates; i++)
        {
            if(vbDiscarded[i])
                continue;

            KeyFrame* pKF = mvpEnoughConsistentCandidates[i];

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            Sim3Solver* pSolver = vpSim3Solvers[i];
            // 最多迭代5次，返回的 Scm 是候选帧pKF到当前帧mpCurrentKF的Sim3变换（T12）
            // m代表候选帧pKF，c代表当前帧
            cv::Mat Scm  = pSolver->iterate(5,bNoMore,vbInliers,nInliers);

            // If Ransac reachs max. iterations discard keyframe
            // 经过n次循环，每次迭代5次，总共迭代 n*5 次
            // 总迭代次数达到最大限制还没有求出合格的Sim3变换，该候选帧剔除
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // If RANSAC returns a Sim3, perform a guided matching and optimize with all correspondences
            if(!Scm.empty())
            {
                // 将vvpMapPointMatches[i]中，inlier存入vpMapPointMatches
                // storge all MapPoints in the loop closing
                vector<MapPoint*> vpMapPointMatches(vvpMapPointMatches[i].size(), static_cast<MapPoint*>(NULL));
                for(size_t j=0, jend=vbInliers.size(); j<jend; j++)
                {
                    // 保存inlier的MapPoint
                    if(vbInliers[j])    // bool
                       vpMapPointMatches[j]=vvpMapPointMatches[i][j];
                }

                // 通过步骤3求取的Sim3变换引导关键帧匹配**弥补步骤2中的漏匹配**
                cv::Mat R = pSolver->GetEstimatedRotation();
                cv::Mat t = pSolver->GetEstimatedTranslation();
                const float s = pSolver->GetEstimatedScale();

                // 查找更多的匹配（成功的闭环匹配需要满足足够多的匹配特征点数，之前使用SearchByBoW进行特征点匹配时会有漏匹配）
                // 通过Sim3变换，确定pKF1的特征点在pKF2中的大致区域，同理，确定pKF2的特征点在pKF1中的大致区域
                // 在该区域内通过描述子进行匹配捕获pKF1和pKF2之前**漏匹配的特征点**，更新匹配 vpMapPointMatches
                matcher.SearchBySim3(mpCurrentKF,pKF,vpMapPointMatches,s,R,t,7.5);

                // gScm表示 候选帧pKF 到当前帧mpCurrentKF的**Sim3变换**
                g2o::Sim3 gScm(Converter::toMatrix3d(R),Converter::toVector3d(t),s);
                // Sim3优化，只要有一个候选帧通过Sim3的求解与优化，就跳出停止对其它候选帧的判断
   	            // vpMapPointMatches 表示mpCurrentKF与 候选帧pKF 的mappoint匹配情况
                const int nInliers = Optimizer::OptimizeSim3(mpCurrentKF, pKF, vpMapPointMatches, gScm, 10, mbFixScale);

                // If optimization is succesful stop ransacs and continue
                if(nInliers>=20)
                {
                    // 表示从候选帧中找到了闭环帧 mpMatchedKF
                    bMatch = true;
                    mpMatchedKF = pKF;
                    // 表示世界坐标系到**候选帧**的Sim3变换
                    g2o::Sim3 gSmw(Converter::toMatrix3d(pKF->GetRotation()),Converter::toVector3d(pKF->GetTranslation()),1.0);
                    // 表示世界坐标系到**当前帧**mpCurrentKF的Sim3变换
                    mg2oScw = gScm*gSmw;
                    // 表示世界坐标系到**当前帧**mpCurrentKF的变换
                    mScw = Converter::toCvMat(mg2oScw);

                    mvpCurrentMatchedPoints = vpMapPointMatches;
                    break;
                }
            }
        }
    }

    if(!bMatch)
    {
        for(int i=0; i<nInitialCandidates; i++)
             mvpEnoughConsistentCandidates[i]->SetErase();
        mpCurrentKF->SetErase();
        return false;
    }

    // Retrieve MapPoints seen in Loop Keyframe and neighbors
    // 将 mpMatchedKF 闭环关键帧相连的**关键帧**全部取出来放入**vpLoopConnectedKFs**
    // 将vpLoopConnectedKFs的**MapPoints**取出来放入 mvpLoopMapPoints
    // all mappoints in the loop closing storged into mvpLoopMapPoints
    vector<KeyFrame*> vpLoopConnectedKFs = mpMatchedKF->GetVectorCovisibleKeyFrames();
    vpLoopConnectedKFs.push_back(mpMatchedKF);
    mvpLoopMapPoints.clear();
    for(vector<KeyFrame*>::iterator vit=vpLoopConnectedKFs.begin(); vit!=vpLoopConnectedKFs.end(); vit++)
    {
        KeyFrame* pKF = *vit;
        vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
        {
            MapPoint* pMP = vpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad() && pMP->mnLoopPointForKF!=mpCurrentKF->mnId)
                {
                    mvpLoopMapPoints.push_back(pMP);
                    // 标记该MapPoint被mpCurrentKF闭环时观测到并添加，避免重复添加
                    pMP->mnLoopPointForKF=mpCurrentKF->mnId;
                }
            }
        }
    }

    // Find more matches projecting with the computed Sim3
    // 将 mvpLoopMapPoints 投影到当前关键帧mpCurrentKF进行**投影匹配**, MapPoint <-> current Frame
    // 根据投影查找**更多的匹配**（成功的闭环匹配需要满足足够多的匹配特征点数）
    // 根据Sim3变换，将每个 mvpLoopMapPoints 投影到mpCurrentKF上，并根据尺度确定一个**搜索区域**，
    // 根据该MapPoint的描述子与该区域内的特征点进行**匹配**，如果匹配误差小于TH_LOW即匹配成功，**更新 mvpCurrentMatchedPoints**
    // mvpCurrentMatchedPoints将用于SearchAndFuse中检测当前帧MapPoints与匹配的MapPoints是否存在**冲突**
    matcher.SearchByProjection(mpCurrentKF, mScw, mvpLoopMapPoints, mvpCurrentMatchedPoints,10);

    // If enough matches accept Loop
    // 判断当前帧与检测出的所有闭环关键帧是否有足够多的MapPoints匹配
    int nTotalMatches = 0;
    for(size_t i=0; i<mvpCurrentMatchedPoints.size(); i++)
    {
        if(mvpCurrentMatchedPoints[i])
            nTotalMatches++;
    }

    // 清空mvpEnoughConsistentCandidates
    if(nTotalMatches>=40)
    {
        for(int i=0; i<nInitialCandidates; i++)
            if(mvpEnoughConsistentCandidates[i]!=mpMatchedKF)
                mvpEnoughConsistentCandidates[i]->SetErase();
        return true;
    }
    else
    {
        for(int i=0; i<nInitialCandidates; i++)
            mvpEnoughConsistentCandidates[i]->SetErase();
        mpCurrentKF->SetErase();
        return false;
    }

}


/**
 * @brief 闭环
 * 1. 通过求解的Sim3以及相对姿态关系，**调整**与当前帧相连的关键帧**位姿**以及这些关键帧观测到的**MapPoints位置**（相连关键帧---当前帧）
 * 2. 将闭环帧以及闭环帧相连的关键帧的MapPoints和与当前帧相连的关键帧的点进行**匹配**（相连关键帧+当前帧---闭环帧+相连关键帧）
 * 3. 通过MapPoints的匹配关系更新这些帧之间的**连接关系**，即更新covisibility graph
 * 4. 对Essential Graph（Pose Graph）进行优化，**MapPoints位置**则根据优化后的位姿做相对应的**调整**
 * 5. 创建线程进行全局Bundle Adjustment
 */
void LoopClosing::CorrectLoop()
{
    cout << "Loop detected!" << endl;

    // Send a stop signal to Local Mapping
    // Avoid new keyframes are inserted while correcting the loop
    // 请求局部地图停止，防止局部地图线程中InsertKeyFrame函数插入新的关键帧
    mpLocalMapper->RequestStop();

    // If a Global Bundle Adjustment is running, abort it
    // 如果正在进行全局BA，放弃它
    if(isRunningGBA())
    {
        unique_lock<mutex> lock(mMutexGBA);
        mbStopGBA = true;

        mnFullBAIdx++;

        if(mpThreadGBA)
        {
            mpThreadGBA->detach();
            delete mpThreadGBA;
        }
    }

    // Wait until Local Mapping has effectively stopped
    // 这里可以利用C++多线程特性修改
    while(!mpLocalMapper->isStopped())
    {
        usleep(1000);
    }

    // Ensure current keyframe is updated
    // 根据共视关系更新当前帧与其它关键帧之间的连接
    mpCurrentKF->UpdateConnections();

    // Retrive keyframes connected to the current keyframe and compute corrected Sim3 pose by propagation
    // 取出与当前帧在covisibility graph连接的关键帧，包括当前关键帧
    mvpCurrentConnectedKFs = mpCurrentKF->GetVectorCovisibleKeyFrames();
    mvpCurrentConnectedKFs.push_back(mpCurrentKF);      // all KeyFrames

    KeyFrameAndPose CorrectedSim3, NonCorrectedSim3;
    // mg2oScw 是根据ComputeSim3()算出来的当前关键帧在世界坐标系中的sim3位姿
    CorrectedSim3[mpCurrentKF]=mg2oScw; // current frame
    cv::Mat Twc = mpCurrentKF->GetPoseInverse();


    {
        // Get Map Mutex
        unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

        // 计算 mvpCurrentConnectedKFs 中各个关键帧（i代表）相对于世界坐标的sim3的**位姿**，其位姿是以mg2oScw为起点，经过位姿传播后的得到的
        // 在将自己本身在tracking时获得的位姿转化为 NonCorrectedSim3
        for(vector<KeyFrame*>::iterator vit=mvpCurrentConnectedKFs.begin(), vend=mvpCurrentConnectedKFs.end(); vit!=vend; vit++)
        {
            KeyFrame* pKFi = *vit;

            cv::Mat Tiw = pKFi->GetPose();  // R, t

            if(pKFi!=mpCurrentKF)
            {
                cv::Mat Tic = Tiw*Twc;
                cv::Mat Ric = Tic.rowRange(0,3).colRange(0,3);
                cv::Mat tic = Tic.rowRange(0,3).col(3);
                g2o::Sim3 g2oSic(Converter::toMatrix3d(Ric),Converter::toVector3d(tic),1.0);
                g2o::Sim3 g2oCorrectedSiw = g2oSic*mg2oScw;
                //Pose corrected with the Sim3 of the loop closure
                CorrectedSim3[pKFi]=g2oCorrectedSiw;
            }

            cv::Mat Riw = Tiw.rowRange(0,3).colRange(0,3);
            cv::Mat tiw = Tiw.rowRange(0,3).col(3);
            g2o::Sim3 g2oSiw(Converter::toMatrix3d(Riw),Converter::toVector3d(tiw),1.0);
            //Pose without correction
            NonCorrectedSim3[pKFi]=g2oSiw;
        }

        // Correct all MapPoints obsrved by current keyframe and neighbors, so that they align with the other side of the loop
        // 遍历 CorrectedSim3 中代表的关键帧，**修正**这些关键帧的MapPoints
        // how to correct MapPoints?
        for(KeyFrameAndPose::iterator mit=CorrectedSim3.begin(), mend=CorrectedSim3.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;
            g2o::Sim3 g2oCorrectedSiw = mit->second;
            g2o::Sim3 g2oCorrectedSwi = g2oCorrectedSiw.inverse();

            g2o::Sim3 g2oSiw =NonCorrectedSim3[pKFi];

            vector<MapPoint*> vpMPsi = pKFi->GetMapPointMatches();
            // 遍历pKFi的mappoint
            for(size_t iMP=0, endMPi = vpMPsi.size(); iMP<endMPi; iMP++)
            {
                MapPoint* pMPi = vpMPsi[iMP];
                if(!pMPi)
                    continue;
                if(pMPi->isBad())
                    continue;
                if(pMPi->mnCorrectedByKF==mpCurrentKF->mnId)
                    continue;

                // Project with non-corrected pose and project back with corrected pose
                cv::Mat P3Dw = pMPi->GetWorldPos();
                Eigen::Matrix<double,3,1> eigP3Dw = Converter::toVector3d(P3Dw);
                Eigen::Matrix<double,3,1> eigCorrectedP3Dw = g2oCorrectedSwi.map(g2oSiw.map(eigP3Dw));

                cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
                pMPi->SetWorldPos(cvCorrectedP3Dw);
                pMPi->mnCorrectedByKF = mpCurrentKF->mnId;
                pMPi->mnCorrectedReference = pKFi->mnId;
                pMPi->UpdateNormalAndDepth();
            }

            // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
            Eigen::Matrix3d eigR = g2oCorrectedSiw.rotation().toRotationMatrix();
            Eigen::Vector3d eigt = g2oCorrectedSiw.translation();
            double s = g2oCorrectedSiw.scale();

            eigt *=(1./s); //[R t/s;0 1]

            cv::Mat correctedTiw = Converter::toCvSE3(eigR,eigt);

            pKFi->SetPose(correctedTiw);

            // Make sure connections are updated
            pKFi->UpdateConnections();
        }

        // Start Loop Fusion
        // Update matched map points and replace if duplicated
        for(size_t i=0; i<mvpCurrentMatchedPoints.size(); i++)
        {
            if(mvpCurrentMatchedPoints[i])
            {
                MapPoint* pLoopMP = mvpCurrentMatchedPoints[i];
                MapPoint* pCurMP = mpCurrentKF->GetMapPoint(i);
                if(pCurMP)
                    pCurMP->Replace(pLoopMP);
                else
                {
                    mpCurrentKF->AddMapPoint(pLoopMP,i);
                    pLoopMP->AddObservation(mpCurrentKF,i);
                    pLoopMP->ComputeDistinctiveDescriptors();
                }
            }
        }

    }

    // Project MapPoints observed in the neighborhood of the loop keyframe
    // into the current keyframe and neighbors using corrected poses.
    // Fuse duplications.
    SearchAndFuse(CorrectedSim3);


    // After the MapPoint fusion, new links in the covisibility graph will appear attaching both sides of the loop
    map<KeyFrame*, set<KeyFrame*> > LoopConnections;

    for(vector<KeyFrame*>::iterator vit=mvpCurrentConnectedKFs.begin(), vend=mvpCurrentConnectedKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;
        vector<KeyFrame*> vpPreviousNeighbors = pKFi->GetVectorCovisibleKeyFrames();

        // Update connections. Detect new links.
        pKFi->UpdateConnections();
        LoopConnections[pKFi]=pKFi->GetConnectedKeyFrames();
        for(vector<KeyFrame*>::iterator vit_prev=vpPreviousNeighbors.begin(), vend_prev=vpPreviousNeighbors.end(); vit_prev!=vend_prev; vit_prev++)
        {
            LoopConnections[pKFi].erase(*vit_prev);
        }
        for(vector<KeyFrame*>::iterator vit2=mvpCurrentConnectedKFs.begin(), vend2=mvpCurrentConnectedKFs.end(); vit2!=vend2; vit2++)
        {
            LoopConnections[pKFi].erase(*vit2);
        }
    }

    // Optimize graph
    Optimizer::OptimizeEssentialGraph(mpMap, mpMatchedKF, mpCurrentKF, NonCorrectedSim3, CorrectedSim3, LoopConnections, mbFixScale);

    mpMap->InformNewBigChange();

    // Add loop edge
    mpMatchedKF->AddLoopEdge(mpCurrentKF);
    mpCurrentKF->AddLoopEdge(mpMatchedKF);

    // Launch a new thread to perform Global Bundle Adjustment
    mbRunningGBA = true;
    mbFinishedGBA = false;
    mbStopGBA = false;
    mpThreadGBA = new thread(&LoopClosing::RunGlobalBundleAdjustment,this,mpCurrentKF->mnId);

    // Loop closed. Release Local Mapping.
    mpLocalMapper->Release();    

    mLastLoopKFid = mpCurrentKF->mnId;   
}

void LoopClosing::SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap)
{
    ORBmatcher matcher(0.8);

    for(KeyFrameAndPose::const_iterator mit=CorrectedPosesMap.begin(), mend=CorrectedPosesMap.end(); mit!=mend;mit++)
    {
        KeyFrame* pKF = mit->first;

        g2o::Sim3 g2oScw = mit->second;
        cv::Mat cvScw = Converter::toCvMat(g2oScw);

        vector<MapPoint*> vpReplacePoints(mvpLoopMapPoints.size(),static_cast<MapPoint*>(NULL));
        matcher.Fuse(pKF,cvScw,mvpLoopMapPoints,4,vpReplacePoints);

        // Get Map Mutex
        unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
        const int nLP = mvpLoopMapPoints.size();
        for(int i=0; i<nLP;i++)
        {
            MapPoint* pRep = vpReplacePoints[i];
            if(pRep)
            {
                pRep->Replace(mvpLoopMapPoints[i]);
            }
        }
    }
}


void LoopClosing::RequestReset()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetRequested = true;
    }

    while(1)
    {
        {
        unique_lock<mutex> lock2(mMutexReset);
        if(!mbResetRequested)
            break;
        }
        usleep(5000);
    }
}

void LoopClosing::ResetIfRequested()
{
    unique_lock<mutex> lock(mMutexReset);
    if(mbResetRequested)
    {
        mlpLoopKeyFrameQueue.clear();
        mLastLoopKFid=0;
        mbResetRequested=false;
    }
}

void LoopClosing::RunGlobalBundleAdjustment(unsigned long nLoopKF)
{
    cout << "Starting Global Bundle Adjustment" << endl;

    int idx =  mnFullBAIdx;
    Optimizer::GlobalBundleAdjustemnt(mpMap,10,&mbStopGBA,nLoopKF,false);

    // Update all MapPoints and KeyFrames
    // Local Mapping was active during BA, that means that there might be new keyframes
    // not included in the Global BA and they are not consistent with the updated map.
    // We need to propagate the correction through the spanning tree
    {
        unique_lock<mutex> lock(mMutexGBA);
        if(idx!=mnFullBAIdx)
            return;

        if(!mbStopGBA)
        {
            cout << "Global Bundle Adjustment finished" << endl;
            cout << "Updating map ..." << endl;
            mpLocalMapper->RequestStop();
            // Wait until Local Mapping has effectively stopped

            while(!mpLocalMapper->isStopped() && !mpLocalMapper->isFinished())
            {
                usleep(1000);
            }

            // Get Map Mutex
            unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

            // Correct keyframes starting at map first keyframe
            list<KeyFrame*> lpKFtoCheck(mpMap->mvpKeyFrameOrigins.begin(),mpMap->mvpKeyFrameOrigins.end());

            while(!lpKFtoCheck.empty())
            {
                KeyFrame* pKF = lpKFtoCheck.front();
                const set<KeyFrame*> sChilds = pKF->GetChilds();
                cv::Mat Twc = pKF->GetPoseInverse();
                for(set<KeyFrame*>::const_iterator sit=sChilds.begin();sit!=sChilds.end();sit++)
                {
                    KeyFrame* pChild = *sit;
                    if(pChild->mnBAGlobalForKF!=nLoopKF)
                    {
                        cv::Mat Tchildc = pChild->GetPose()*Twc;
                        pChild->mTcwGBA = Tchildc*pKF->mTcwGBA;//*Tcorc*pKF->mTcwGBA;
                        pChild->mnBAGlobalForKF=nLoopKF;

                    }
                    lpKFtoCheck.push_back(pChild);
                }

                pKF->mTcwBefGBA = pKF->GetPose();
                pKF->SetPose(pKF->mTcwGBA);
                lpKFtoCheck.pop_front();
            }

            // Correct MapPoints
            const vector<MapPoint*> vpMPs = mpMap->GetAllMapPoints();

            for(size_t i=0; i<vpMPs.size(); i++)
            {
                MapPoint* pMP = vpMPs[i];

                if(pMP->isBad())
                    continue;

                if(pMP->mnBAGlobalForKF==nLoopKF)
                {
                    // If optimized by Global BA, just update
                    pMP->SetWorldPos(pMP->mPosGBA);
                }
                else
                {
                    // Update according to the correction of its reference keyframe
                    KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();

                    if(pRefKF->mnBAGlobalForKF!=nLoopKF)
                        continue;

                    // Map to non-corrected camera
                    cv::Mat Rcw = pRefKF->mTcwBefGBA.rowRange(0,3).colRange(0,3);
                    cv::Mat tcw = pRefKF->mTcwBefGBA.rowRange(0,3).col(3);
                    cv::Mat Xc = Rcw*pMP->GetWorldPos()+tcw;

                    // Backproject using corrected camera
                    cv::Mat Twc = pRefKF->GetPoseInverse();
                    cv::Mat Rwc = Twc.rowRange(0,3).colRange(0,3);
                    cv::Mat twc = Twc.rowRange(0,3).col(3);

                    pMP->SetWorldPos(Rwc*Xc+twc);
                }
            }            

            mpMap->InformNewBigChange();

            mpLocalMapper->Release();

            cout << "Map updated!" << endl;
        }

        mbFinishedGBA = true;
        mbRunningGBA = false;
    }
}

void LoopClosing::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool LoopClosing::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LoopClosing::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
}

bool LoopClosing::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}


} //namespace ORB_SLAM
