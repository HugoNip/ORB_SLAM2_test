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

#include "KeyFrame.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include<mutex>

namespace ORB_SLAM2
{

long unsigned int KeyFrame::nNextId=0;

// F: 当前帧
KeyFrame::KeyFrame(Frame &F, Map *pMap, KeyFrameDatabase *pKFDB):
    mnFrameId(F.mnId),  mTimeStamp(F.mTimeStamp), 
    mnGridCols(FRAME_GRID_COLS), mnGridRows(FRAME_GRID_ROWS),
    mfGridElementWidthInv(F.mfGridElementWidthInv), 
    mfGridElementHeightInv(F.mfGridElementHeightInv),
    mnTrackReferenceForFrame(0), mnFuseTargetForKF(0), 
    mnBALocalForKF(0), mnBAFixedForKF(0),
    mnLoopQuery(0), mnLoopWords(0), 
    mnRelocQuery(0), mnRelocWords(0), 
    mnBAGlobalForKF(0),
    fx(F.fx), fy(F.fy), cx(F.cx), cy(F.cy), invfx(F.invfx), invfy(F.invfy),
    mbf(F.mbf), mb(F.mb), mThDepth(F.mThDepth), N(F.N), 
    mvKeys(F.mvKeys), mvKeysUn(F.mvKeysUn),
    mvuRight(F.mvuRight), mvDepth(F.mvDepth), 
    mDescriptors(F.mDescriptors.clone()),
    mBowVec(F.mBowVec), mFeatVec(F.mFeatVec), 
    mnScaleLevels(F.mnScaleLevels), mfScaleFactor(F.mfScaleFactor),
    mfLogScaleFactor(F.mfLogScaleFactor), mvScaleFactors(F.mvScaleFactors), 
    mvLevelSigma2(F.mvLevelSigma2), mvInvLevelSigma2(F.mvInvLevelSigma2), 
    mnMinX(F.mnMinX), mnMinY(F.mnMinY), mnMaxX(F.mnMaxX), mnMaxY(F.mnMaxY), 
    mK(F.mK), mvpMapPoints(F.mvpMapPoints), mpKeyFrameDB(pKFDB),
    mpORBvocabulary(F.mpORBvocabulary), mbFirstConnection(true), mpParent(NULL), 
    mbNotErase(false), mbToBeErased(false), mbBad(false), 
    mHalfBaseline(F.mb/2), mpMap(pMap)
{
    //将下一帧的帧号赋值给mnId，然后自增1
    mnId=nNextId++;

    //根据栅格的列数重置栅格的size
    mGrid.resize(mnGridCols);
    //将该帧的栅格内信息拷贝了一份给关键帧类内的变量
    for(int i=0; i<mnGridCols;i++)
    {
        mGrid[i].resize(mnGridRows);
        for(int j=0; j<mnGridRows; j++)
            mGrid[i][j] = F.mGrid[i][j];
    }

    //最后将当前帧的姿态赋给该关键帧
    SetPose(F.mTcw);
}

void KeyFrame::ComputeBoW()
{
    if(mBowVec.empty() || mFeatVec.empty())
    {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        // Feature vector associate features with nodes in the 4th level (from leaves up)
        // We assume the vocabulary tree has 6 levels, change the 4 otherwise
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
}

void KeyFrame::SetPose(const cv::Mat &Tcw_)
{
    unique_lock<mutex> lock(mMutexPose);
    Tcw_.copyTo(Tcw);
    cv::Mat Rcw = Tcw.rowRange(0,3).colRange(0,3);
    cv::Mat tcw = Tcw.rowRange(0,3).col(3);
    cv::Mat Rwc = Rcw.t();
    Ow = -Rwc*tcw; // camera center

    Twc = cv::Mat::eye(4,4,Tcw.type());
    Rwc.copyTo(Twc.rowRange(0,3).colRange(0,3));
    Ow.copyTo(Twc.rowRange(0,3).col(3));
    // center为相机坐标系（左目）下，立体相机 中心 的坐标
    // 立体相机 中心点 坐标与左目相机坐标之间只是在x轴上相差 mHalfBaseline ,
    // 因此可以看出，立体相机中两个摄像头的连线为x轴，正方向为左目相机指向右目相机
    cv::Mat center = (cv::Mat_<float>(4,1) << mHalfBaseline, 0 , 0, 1);
    Cw = Twc*center;
}

cv::Mat KeyFrame::GetPose()
{
    unique_lock<mutex> lock(mMutexPose);
    return Tcw.clone();
}

cv::Mat KeyFrame::GetPoseInverse()
{
    unique_lock<mutex> lock(mMutexPose);
    return Twc.clone();
}

cv::Mat KeyFrame::GetCameraCenter()
{
    unique_lock<mutex> lock(mMutexPose);
    return Ow.clone();
}

cv::Mat KeyFrame::GetStereoCenter()
{
    unique_lock<mutex> lock(mMutexPose);
    return Cw.clone();
}


cv::Mat KeyFrame::GetRotation()
{
    unique_lock<mutex> lock(mMutexPose);
    return Tcw.rowRange(0,3).colRange(0,3).clone();
}

cv::Mat KeyFrame::GetTranslation()
{
    unique_lock<mutex> lock(mMutexPose);
    return Tcw.rowRange(0,3).col(3).clone();
}


// pKF     需要关联的关键帧
// weight  权重，即该关键帧与pKF共同观测到的3d点数量
void KeyFrame::AddConnection(KeyFrame *pKF, const int &weight)
{
    {
        unique_lock<mutex> lock(mMutexConnections);

        // std::map::count函数只可能返回0或1两种情况
        // 此处0表示之前没有过连接，1表示有过连接
        
        if(!mConnectedKeyFrameWeights.count(pKF))
            // 之前没有连接时，要用权重赋值，即添加连接
            mConnectedKeyFrameWeights[pKF]=weight; // map(match id, weight)
        else if(mConnectedKeyFrameWeights[pKF]!=weight)
            // 有连接，但权重发生变化时，也要用权重赋值，即更新权重
            mConnectedKeyFrameWeights[pKF]=weight;
        else
            return;
    }

    // 更新最好的Covisibility
    UpdateBestCovisibles();
}


void KeyFrame::UpdateBestCovisibles()
{
    unique_lock<mutex> lock(mMutexConnections);
    vector<pair<int,KeyFrame*> > vPairs;
    vPairs.reserve(mConnectedKeyFrameWeights.size());
    // 取出所有连接的关键帧，将元素取出放入一个pair组成的vector中，排序后放入vPairs
    // mConnectedKeyFrameWeights 的类型为 std::map<KeyFrame*,int>，
    // 而 vPairs 变量将共视的3D点数放在前面，利于排序
    for(map<KeyFrame*,int>::iterator mit=mConnectedKeyFrameWeights.begin(), 
        mend=mConnectedKeyFrameWeights.end(); mit!=mend; mit++)
       vPairs.push_back(make_pair(mit->second,mit->first));

    // 按照权重进行排序
    sort(vPairs.begin(),vPairs.end());
    list<KeyFrame*> lKFs;
    list<int> lWs;
    for(size_t i=0, iend=vPairs.size(); i<iend;i++)
    {
        // 所以定义的链表中权重由大到小排列 要用 push_front
        lKFs.push_front(vPairs[i].second);
        lWs.push_front(vPairs[i].first);
    }

    // 更新排序好的连接关键帧及其对应的权重
    mvpOrderedConnectedKeyFrames = vector<KeyFrame*>(lKFs.begin(),lKFs.end());
    mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());    
}


set<KeyFrame*> KeyFrame::GetConnectedKeyFrames()
{
    unique_lock<mutex> lock(mMutexConnections);
    set<KeyFrame*> s;
    for(map<KeyFrame*,int>::iterator mit=mConnectedKeyFrameWeights.begin();
        mit!=mConnectedKeyFrameWeights.end();mit++)
        s.insert(mit->first);
    return s;
}


vector<KeyFrame*> KeyFrame::GetVectorCovisibleKeyFrames()
{
    unique_lock<mutex> lock(mMutexConnections);
    return mvpOrderedConnectedKeyFrames;
}


vector<KeyFrame*> KeyFrame::GetBestCovisibilityKeyFrames(const int &N)
{
    unique_lock<mutex> lock(mMutexConnections);
    if((int)mvpOrderedConnectedKeyFrames.size()<N)
        return mvpOrderedConnectedKeyFrames;
    else
        return vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(),
                                 mvpOrderedConnectedKeyFrames.begin()+N); // Orderly
}


vector<KeyFrame*> KeyFrame::GetCovisiblesByWeight(const int &w)
{
    unique_lock<mutex> lock(mMutexConnections);

    if(mvpOrderedConnectedKeyFrames.empty())
        return vector<KeyFrame*>();

    vector<int>::iterator it = upper_bound(mvOrderedWeights.begin(),
        mvOrderedWeights.end(),w,KeyFrame::weightComp);

    if(it==mvOrderedWeights.end())
        return vector<KeyFrame*>();
    else
    {
        int n = it-mvOrderedWeights.begin();
        return vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(), 
                                 mvpOrderedConnectedKeyFrames.begin()+n);
    }
}


int KeyFrame::GetWeight(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexConnections);
    if(mConnectedKeyFrameWeights.count(pKF))
        return mConnectedKeyFrameWeights[pKF];
    else
        return 0;
}


// MapPoint相关
// 这一类函数的内容同样比较简单，主要围绕存放MapPoint的容器mvpMapPoints进行
void KeyFrame::AddMapPoint(MapPoint *pMP, const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mvpMapPoints[idx]=pMP;  // idx of keypoint <-> mappoint
}


void KeyFrame::EraseMapPointMatch(const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mvpMapPoints[idx]=static_cast<MapPoint*>(NULL);
}


void KeyFrame::EraseMapPointMatch(MapPoint* pMP)
{
    int idx = pMP->GetIndexInKeyFrame(this);
    if(idx>=0)
        mvpMapPoints[idx]=static_cast<MapPoint*>(NULL);
}


/**
 * same idx, replace with new pMP
 */
void KeyFrame::ReplaceMapPointMatch(const size_t &idx, MapPoint* pMP)
{
    mvpMapPoints[idx]=pMP;
}


// 得到map point不是bad的mappoint 集合
// 
set<MapPoint*> KeyFrame::GetMapPoints()
{
    unique_lock<mutex> lock(mMutexFeatures);
    set<MapPoint*> s;
    for(size_t i=0, iend=mvpMapPoints.size(); i<iend; i++)
    {
        if(!mvpMapPoints[i])
            continue;
        MapPoint* pMP = mvpMapPoints[i];
        if(!pMP->isBad())   // 得到map point不是bad的mappoint 集合
            s.insert(pMP);
    }
    return s;
}


// 获取被观测相机数大于等于minObs的MapPoint
int KeyFrame::TrackedMapPoints(const int &minObs)
{
    unique_lock<mutex> lock(mMutexFeatures);

    int nPoints=0;
    const bool bCheckObs = minObs>0;
    for(int i=0; i<N; i++)
    {
        MapPoint* pMP = mvpMapPoints[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                if(bCheckObs)
                {
                    if(mvpMapPoints[i]->Observations()>=minObs)
                        nPoints++;
                }
                else
                    nPoints++;
            }
        }
    }

    return nPoints;
}


// all matched mappoint, including bad mappoint
vector<MapPoint*> KeyFrame::GetMapPointMatches()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpMapPoints;
}


// get one mappoint
MapPoint* KeyFrame::GetMapPoint(const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpMapPoints[idx];
}


/**
 * 该函数主要包含以下三部分内容：
 * a. 首先获得该KF的**所有MapPoint**，然后遍历观测到这些3d点的其它所有KFs, 对每一个找到的KF，先存储到相应的容器中
 * b. 计算所有共视帧与该帧的**连接权重**，权重 即为 共视的3d点的 数量，对这些连接按照权重从大到小进行排序。
 *    当该权重必须大于一个阈值，便 在两帧之间 建立边，如果没有超过该阈值的权重，那么就只保留 权重最大的 边
 *   （与其它关键帧的 共视程度 比较高）
 *    edge is built between the KeyFrames with the most shared MapPoints
 * c. **更新 covisibility graph**，即把计算的 边 用来给 图 赋值，然后设置 spanning tree 中该帧的 父节点 ，
 *    即共视程度最高的那一帧
 */
void KeyFrame::UpdateConnections()
{
    // 在没有执行这个函数前，关键帧只和 MapPoints 之间有 连接关系，这个函数可以更新关键帧之间的连接关系

    //===============对应a部分内容==================================
    map<KeyFrame*,int> KFcounter;   // 和此关键帧有共视关系的**关键帧**及其可以共视的mappoint**数量**

    vector<MapPoint*> vpMP;         // 此关键帧可以看到的mappoint

    {
        // 获得该 关键帧 的所有3D点
        unique_lock<mutex> lockMPs(mMutexFeatures);
        vpMP = mvpMapPoints; // MapPoints container for the KF
    }

    // For all map points in one keyframe, check in which other keyframes are they seen
    // Increase counter for those keyframes
    // 即 统计 每一个关键帧 都有多少 关键帧 与 它存在 共视关系，统计结果放在 KFcounter
    // 计算每一个关键帧都有多少其他关键帧与它存在共视关系，结果放在KFcounter
    // 遍历此关键帧可以看到的mappoint
    for(vector<MapPoint*>::iterator vit=vpMP.begin(), vend=vpMP.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;

        if(!pMP)
            continue;

        if(pMP->isBad())
            continue;

        // 对于每一个 MapPoint ， observations 记录了可以观测到该 MapPoint 的所有 关键帧
        map<KeyFrame*,size_t> observations = pMP->GetObservations();

        // 遍历此mappoint可以被看到的所有关键帧
        for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            // 除去自身，自己与自己不算共视
            if(mit->first->mnId==mnId)
                continue;
            KFcounter[mit->first]++;    // number/weight of covisiable KeyFrames for one KeyFrame for one MapPoint
                                        // KFcounter[KF] = weight
        }
    }

    // This should not happen
    // 没有其他关键帧和此关键帧有关系
    if(KFcounter.empty())
        return;

    //===============对应b部分内容==================================
    // If the counter is greater than threshold add connection between two keyframes
    // In case no keyframe counter is over threshold add the one with maximum counter
    // 通过3D点间接统计可以观测到这些3D点的所有关键帧之间的共视程度
    int nmax=0;
    KeyFrame* pKFmax=NULL;
    int th = 15;

    // vPairs 记录 与其它关键帧共视帧数大于th的 关键帧
    // pair<weight,KeyFrame> 将关键帧的权重写在前面，关键帧写在后面方便后面排序
    // 当共视mappoint点数量达到一定阈值th的情况下，在covisibility graph添加一条边
    vector<pair<int,KeyFrame*> > vPairs;
    vPairs.reserve(KFcounter.size());
    for(map<KeyFrame*,int>::iterator mit=KFcounter.begin(), mend=KFcounter.end(); mit!=mend; mit++) 
    {
        if(mit->second>nmax) // weight > nmax
        {
            nmax=mit->second;
            // 找到对应权重最大的关键帧（共视程度最高的关键帧）
            pKFmax=mit->first;
        }
        if(mit->second>=th) // weight > th
        {
            // 对应权重 需要大于 阈值 ，对这些关键帧建立连接
            // pair<weight, KF>
            vPairs.push_back(make_pair(mit->second, mit->first));
            // 更新 KFcounter 中该关键帧的 mConnectedKeyFrameWeights
            // 更新其它 KeyFrame 的 mConnectedKeyFrameWeights ，更新其它关键帧与当前帧的连接权重
            (mit->first)->AddConnection(this, mit->second);
        }
    }

    // 如果没有超过阈值的权重，则对权重最大的关键帧建立连接
    if(vPairs.empty())
    {
        // 如果每个关键帧与它共视的关键帧的个数都少于 th ，
        // 那就只更新与其它关键帧共视程度最高的关键帧的 mConnectedKeyFrameWeights
        // 这是对之前 th 这个阈值可能过高的一个补丁
        vPairs.push_back(make_pair(nmax, pKFmax));
        pKFmax->AddConnection(this,nmax);
    }

    // vPairs 里存的都是 相互共视程度 比较高 的关键帧和共视权重，由大到小
    sort(vPairs.begin(),vPairs.end());
    list<KeyFrame*> lKFs;
    list<int> lWs;
    // 把权重大的放在前面，这样
    for(size_t i=0; i<vPairs.size();i++)
    {
        lKFs.push_front(vPairs[i].second);
        lWs.push_front(vPairs[i].first);
    }

    //===============对应c部分内容==================================
    {
        unique_lock<mutex> lockCon(mMutexConnections);

        // mspConnectedKeyFrames = spConnectedKeyFrames;
        // 更新共视图covisibility graph的连接(weight)
        mConnectedKeyFrameWeights = KFcounter;

        // 更新covisibility graph连接
	    // mvpOrderedConnectedKeyFrames 权重由大到小排列
        mvpOrderedConnectedKeyFrames = vector<KeyFrame*>(lKFs.begin(),lKFs.end());
        mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());

        // 更新生成树的连接
        // 如果不是第一个关键帧，且此节点没有父节点
        if(mbFirstConnection && mnId!=0)
        {
            // 初始化 该关键帧的 父关键帧 为 **共视程度最高** 的那个关键帧
            mpParent = mvpOrderedConnectedKeyFrames.front();
            // 建立双向连接关系
            // 共视程度最高的那个关键帧在Spanning Tree中的子节点设置为此关键帧
            mpParent->AddChild(this);
            mbFirstConnection = false;
        }

    }
}


// spanning tree 相关
// 这一类函数所有的操作都是在围绕自己的子节点和父节点，其中子节点可能有多个，
// 所以是一个容器 mspChildrens ，父节点只能有一个，所以是个变量mpParent。
void KeyFrame::AddChild(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mspChildrens.insert(pKF);
}


void KeyFrame::EraseChild(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mspChildrens.erase(pKF);
}


void KeyFrame::ChangeParent(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mpParent = pKF;
    pKF->AddChild(this);
}


set<KeyFrame*> KeyFrame::GetChilds()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspChildrens;
}


KeyFrame* KeyFrame::GetParent()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mpParent;
}


bool KeyFrame::hasChild(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspChildrens.count(pKF);
}


void KeyFrame::AddLoopEdge(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mbNotErase = true;
    mspLoopEdges.insert(pKF);
}


set<KeyFrame*> KeyFrame::GetLoopEdges()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspLoopEdges;
}


void KeyFrame::SetNotErase()
{
    unique_lock<mutex> lock(mMutexConnections);
    mbNotErase = true;
}


void KeyFrame::SetErase()
{
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(mspLoopEdges.empty())
        {
            mbNotErase = false;
        }
    }

    if(mbToBeErased)
    {
        SetBadFlag();
    }
}


/**
 * 需要删除的是该关键帧和其他所有帧、地图点之间的连接关系，但是删除会带来一个问题，
 * 就是它可能是其他节点的父节点，在删除之前需要告诉自己所有的子节点，换个 爸爸 ，
 * 这个函数里绝大部分代码都是在完成这一步。
 * 借鉴博客链接：https://blog.csdn.net/weixin_39373577/article/details/85226187
 * 
 * 步骤一：遍历所有和当前关键帧共视的关键帧，删除他们与当前关键帧的联系。
 * 步骤二：遍历每一个当前关键帧的地图点，删除每一个地图点和当前关键帧的联系。
 * 步骤三：清空和当前关键帧的共视关键帧集合和带顺序的关键帧集合。
 * 步骤四：共视图更新完毕后，还需要更新生成树。
 *        这个比较难理解。。。真实删除当前关键帧之前，需要处理好父亲和儿子关键帧关系，
 *        不然会造成整个关键帧维护的图断裂，或者混乱，不能够为后端提供较好的初值
 *      （理解起来就是父亲挂了，儿子需要找新的父亲，在候选父亲里找，当前帧的父亲肯定在候选父亲中）。
 * 步骤五：遍历所有把当前关键帧当成父关键帧的子关键帧。重新为他们找父关键帧。
 *        设置一个候选父关键帧集合（集合里包含了当前帧的父帧和子帧？）
 * 步骤六：对于每一个子关键帧，找到与它共视的关键帧集合，遍历它，看看是否有候选父帧集合里的帧，
 *        如果有，就把这个帧当做新的父帧。
 * 步骤七：如果有子关键帧没有找到新的父帧，那么直接把当前帧的父帧（爷）当成它的父帧
 */
void KeyFrame::SetBadFlag()
{   
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(mnId==0)
            return;
        else if(mbNotErase)
        {
            mbToBeErased = true;
            return;
        }
    }

    for(map<KeyFrame*,int>::iterator mit = mConnectedKeyFrameWeights.begin(), 
        mend=mConnectedKeyFrameWeights.end(); mit!=mend; mit++)
        mit->first->EraseConnection(this);

    for(size_t i=0; i<mvpMapPoints.size(); i++)
        if(mvpMapPoints[i])
            mvpMapPoints[i]->EraseObservation(this);
    {
        unique_lock<mutex> lock(mMutexConnections);
        unique_lock<mutex> lock1(mMutexFeatures);

        mConnectedKeyFrameWeights.clear();
        mvpOrderedConnectedKeyFrames.clear();

        // Update Spanning Tree
        set<KeyFrame*> sParentCandidates;
        sParentCandidates.insert(mpParent);

        // Assign at each iteration one children with a parent 
        // (the pair with highest covisibility weight)
        // Include that children as new parent candidate for the rest
        while(!mspChildrens.empty())
        {
            bool bContinue = false;

            int max = -1;
            KeyFrame* pC;
            KeyFrame* pP;

            for(set<KeyFrame*>::iterator sit=mspChildrens.begin(), 
                send=mspChildrens.end(); sit!=send; sit++)
            {
                KeyFrame* pKF = *sit;
                if(pKF->isBad())
                    continue;

                // Check if a parent candidate is connected to the keyframe
                vector<KeyFrame*> vpConnected = pKF->GetVectorCovisibleKeyFrames();
                for(size_t i=0, iend=vpConnected.size(); i<iend; i++)
                {
                    for(set<KeyFrame*>::iterator spcit=sParentCandidates.begin(), 
                        spcend=sParentCandidates.end(); spcit!=spcend; spcit++)
                    {
                        if(vpConnected[i]->mnId == (*spcit)->mnId)
                        {
                            int w = pKF->GetWeight(vpConnected[i]);
                            if(w>max)
                            {
                                pC = pKF;
                                pP = vpConnected[i];
                                max = w;
                                bContinue = true;
                            }
                        }
                    }
                }
            }

            if(bContinue)
            {
                pC->ChangeParent(pP);
                sParentCandidates.insert(pC);
                mspChildrens.erase(pC);
            }
            else
                break;
        }

        // If a children has no covisibility links with any parent candidate, 
        // assign to the original parent of this KF
        if(!mspChildrens.empty())
            for(set<KeyFrame*>::iterator sit=mspChildrens.begin(); 
                sit!=mspChildrens.end(); sit++)
            {
                (*sit)->ChangeParent(mpParent);
            }

        mpParent->EraseChild(this);
        mTcp = Tcw*mpParent->GetPoseInverse();
        mbBad = true;
    }


    mpMap->EraseKeyFrame(this);
    mpKeyFrameDB->erase(this);
}


bool KeyFrame::isBad()
{
    unique_lock<mutex> lock(mMutexConnections);
    return mbBad;
}


// 清除一个关键帧与其他帧对应的边
void KeyFrame::EraseConnection(KeyFrame* pKF)
{
    bool bUpdate = false;
    {
        unique_lock<mutex> lock(mMutexConnections);
        // 如果当前帧有连接关系，则删除
        if(mConnectedKeyFrameWeights.count(pKF))
        {
            mConnectedKeyFrameWeights.erase(pKF);
            bUpdate=true;
        }
    }

    // 如果删除了连接关系，便需要重新对权重进行排序
    if(bUpdate)
        UpdateBestCovisibles();
}


vector<size_t> KeyFrame::GetFeaturesInArea(const float &x, const float &y, const float &r) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    const int nMinCellX = max(0,(int)floor((x-mnMinX-r)*mfGridElementWidthInv));
    if(nMinCellX>=mnGridCols)
        return vIndices;

    const int nMaxCellX = min((int)mnGridCols-1,(int)ceil((x-mnMinX+r)*mfGridElementWidthInv));
    if(nMaxCellX<0)
        return vIndices;

    const int nMinCellY = max(0,(int)floor((y-mnMinY-r)*mfGridElementHeightInv));
    if(nMinCellY>=mnGridRows)
        return vIndices;

    const int nMaxCellY = min((int)mnGridRows-1,(int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t> vCell = mGrid[ix][iy];
            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}


bool KeyFrame::IsInImage(const float &x, const float &y) const
{
    return (x>=mnMinX && x<mnMaxX && y>=mnMinY && y<mnMaxY);
}


cv::Mat KeyFrame::UnprojectStereo(int i)
{
    const float z = mvDepth[i];
    if(z>0)
    {
        const float u = mvKeys[i].pt.x;
        const float v = mvKeys[i].pt.y;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);

        unique_lock<mutex> lock(mMutexPose);
        return Twc.rowRange(0,3).colRange(0,3)*x3Dc+Twc.rowRange(0,3).col(3);
    }
    else
        return cv::Mat();
}


/**
 * 计算场景中的中位深度(median depth)
 * 
 * 步骤一：获取每个 地图点 的 世界位姿
 * 步骤二：找出当前帧 Z方向上的 旋转 和 平移 ，求每个地图点在当前相机坐标系中的 z轴位置 ，求 平均值
 */
float KeyFrame::ComputeSceneMedianDepth(const int q)
{
    vector<MapPoint*> vpMapPoints;
    cv::Mat Tcw_;
    {
        unique_lock<mutex> lock(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPose);
        vpMapPoints = mvpMapPoints;
        Tcw_ = Tcw.clone();
    }

    vector<float> vDepths;
    vDepths.reserve(N);
    cv::Mat Rcw2 = Tcw_.row(2).colRange(0,3);
    Rcw2 = Rcw2.t();
    float zcw = Tcw_.at<float>(2,3);
    for(int i=0; i<N; i++)
    {
        if(mvpMapPoints[i])
        {
            MapPoint* pMP = mvpMapPoints[i];
            cv::Mat x3Dw = pMP->GetWorldPos();
            float z = Rcw2.dot(x3Dw)+zcw;
            vDepths.push_back(z);
        }
    }

    sort(vDepths.begin(),vDepths.end());

    return vDepths[(vDepths.size()-1)/q];
}

} //namespace ORB_SLAM
