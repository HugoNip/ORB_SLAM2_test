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

#include "MapPoint.h"
#include "ORBmatcher.h"

#include<mutex>

namespace ORB_SLAM2
{

long unsigned int MapPoint::nNextId=0;
mutex MapPoint::mGlobalMutex;

MapPoint::MapPoint(const cv::Mat &Pos, KeyFrame *pRefKF, Map* pMap):
    mnFirstKFid(pRefKF->mnId), mnFirstFrame(pRefKF->mnFrameId), nObs(0), mnTrackReferenceForFrame(0),
    mnLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopPointForKF(0), mnCorrectedByKF(0),
    mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(pRefKF), mnVisible(1), mnFound(1), mbBad(false),
    mpReplaced(static_cast<MapPoint*>(NULL)), mfMinDistance(0), mfMaxDistance(0), mpMap(pMap)
{
    Pos.copyTo(mWorldPos);
    mNormalVector = cv::Mat::zeros(3,1,CV_32F);

    // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
    unique_lock<mutex> lock(mpMap->mMutexPointCreation);
    mnId=nNextId++;
}


/**
 * @brief
 * get the information of normal vector
 */
MapPoint::MapPoint(const cv::Mat &Pos, Map* pMap, Frame* pFrame, const int &idxF):
    mnFirstKFid(-1), mnFirstFrame(pFrame->mnId), nObs(0), mnTrackReferenceForFrame(0), mnLastFrameSeen(0),
    mnBALocalForKF(0), mnFuseCandidateForKF(0),mnLoopPointForKF(0), mnCorrectedByKF(0),
    mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(static_cast<KeyFrame*>(NULL)), mnVisible(1),
    mnFound(1), mbBad(false), mpReplaced(NULL), mpMap(pMap)
{
    Pos.copyTo(mWorldPos);
    cv::Mat Ow = pFrame->GetCameraCenter();
    mNormalVector = mWorldPos - Ow;         // mWorldPos: Position in absolute coordinates
    mNormalVector = mNormalVector/cv::norm(mNormalVector);

    cv::Mat PC = Pos - Ow;
    const float dist = cv::norm(PC);
    const int level = pFrame->mvKeysUn[idxF].octave;
    const float levelScaleFactor =  pFrame->mvScaleFactors[level];
    const int nLevels = pFrame->mnScaleLevels;

    mfMaxDistance = dist*levelScaleFactor;
    mfMinDistance = mfMaxDistance/pFrame->mvScaleFactors[nLevels-1];

    pFrame->mDescriptors.row(idxF).copyTo(mDescriptor);

    // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
    unique_lock<mutex> lock(mpMap->mMutexPointCreation);
    mnId=nNextId++;
}

void MapPoint::SetWorldPos(const cv::Mat &Pos)
{
    unique_lock<mutex> lock2(mGlobalMutex);
    unique_lock<mutex> lock(mMutexPos);
    Pos.copyTo(mWorldPos);
}

cv::Mat MapPoint::GetWorldPos()
{
    unique_lock<mutex> lock(mMutexPos);
    return mWorldPos.clone();
}


/**
 * @brief
 * GetNormal vector of MapPoint
 */
cv::Mat MapPoint::GetNormal()
{
    unique_lock<mutex> lock(mMutexPos);
    return mNormalVector.clone();
}

KeyFrame* MapPoint::GetReferenceKeyFrame()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mpRefKF;
}


/**
 * storage the Observation for a MapPoint
 * storage the keyframe and idx in the keyframe
 * 
 * 它的作用是判断此 关键帧 是否已经在观测关系中了，如果是，这里就不会添加； 
 * 如果不是，往下 记录下 此关键帧 以及此 MapPoint 的 idx，就算是记录下观测信息了
 * 
 * @param pKF   new KeyFrame
 * @param idx   idx of old MapPoint in keyframe, 该地图点在关键帧中对应的索引值
 * 
 * @note pMP->AddObservation(pKF, mit->second);
 */
void MapPoint::AddObservation(KeyFrame* pKF, size_t idx)
{
    unique_lock<mutex> lock(mMutexFeatures);

    // whether or not this keyframe has been in current observatons
    // std::map<KeyFrame*, int> mObservations 
    // 它是用来存放 MapPoint 观测关系 的容器，把能够观测到该 MapPoint 的 关键帧，
    // 以及 MapPoint 在该关键帧中对应的 索引值 关联并存储起来
    if(mObservations.count(pKF))
        return;
    
    // add this keyframe into mObservations, use old idx of MapPoint
    // it means replace the old keyframe
    mObservations[pKF]=idx;

    // const std::vector<float> mvuRight; // negative value for monocular points
    // nObs: 它用来记录被观测的次数
    if(pKF->mvuRight[idx]>=0)
        nObs+=2; // stereo
    else
        nObs++; // monocular
}


/**
 * For a mappoint, 
 * if it is not observed by one keyframe, 
 * erase this keyframe, and idx
 * mObservations->first: KF
 * mObservations->second: idx
 * so, erase KF is OK
 * 
 * 首先,判断该 关键帧 是否在 观测 中，如果在，就从存放观测关系的容器 mObservations 中移除该关键帧，
 * 接着,判断该帧是否是 参考关键帧 ，如果是， 参考关键帧 换成观测的第一帧，因为不能没有 参考关键帧
 * 删除以后，如果该MapPoint被观测的次数小于2，那么这个MapPoint就没有存在的必要了，需要删除
 * 
 * @param pKF   关键帧
 */
void MapPoint::EraseObservation(KeyFrame* pKF)
{
    bool bBad=false;
    {
        unique_lock<mutex> lock(mMutexFeatures);
        // whether or not this keyframe (pKF) has been in current observatons
        if(mObservations.count(pKF))
        {
            int idx = mObservations[pKF]; // index of mappoint
            // delete number of observations for this mappoint
            if(pKF->mvuRight[idx]>=0)
                nObs-=2;
            else
                nObs--;

            mObservations.erase(pKF);

            /**
             * whether or not this keyframe is Reference KeyFrame
             * if true, reset Reference KeyFrame again
             * if not reset, after deleting this keyFrame,
             * there is no Reference KeyFrame
             */
            if(mpRefKF==pKF)
                mpRefKF=mObservations.begin()->first;

            // If only 2 observations or less, discard point
            if(nObs<=2)
                bBad=true;
        }
    }

    if(bBad)
        SetBadFlag();
}

map<KeyFrame*, size_t> MapPoint::GetObservations()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mObservations;
}

int MapPoint::Observations()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return nObs;    // how many times that this MapPoint can be seen by keyframes
}


/**
 * 1. set bad flag
 * 2. clear all observations for this mappoint 
 * 
 * 如果该MapPoint被观测的次数小于2，那么这个MapPoint就没有存在的必要了，需要删除
 */
void MapPoint::SetBadFlag()
{
    map<KeyFrame*,size_t> obs;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        mbBad=true;
        obs = mObservations;
        // clear all observations for this mappoint
        mObservations.clear();
    }
    for(map<KeyFrame*,size_t>::iterator mit=obs.begin(), mend=obs.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;
        pKF->EraseMapPointMatch(mit->second);
    }

    mpMap->EraseMapPoint(this);
}

MapPoint* MapPoint::GetReplaced()
{
    unique_lock<mutex> lock1(mMutexFeatures);
    unique_lock<mutex> lock2(mMutexPos);
    return mpReplaced;
}


/**
 * @brief
 * 该函数的作用是将当前 MapPoint(this)，替换成 pMP，
 * 这主要是因为在使用闭环时，完成 闭环优化 以后，需要 调整 MapPoint 和 KeyFrame，建立新的关系
 * 具体流程是循环遍历所有的 observation/KeyFrame ，判断此 MapPoint 是否在 该KeyFrame 中，
 * 如果在，那么只要移除原来 MapPoint 的匹配信息，最后增加这个 MapPoint 找到的数量 以及 可见的次数，
 * 另外地图中要移除原来的那个 MapPoint 
 * 最后需要计算这个点独有的描述子
 * 
 * @param pMP   就是用来替换的地图点
 */
void MapPoint::Replace(MapPoint* pMP)
{
    // if the same one
    // 如果传入的该 MapPoint 就是当前的 MapPoint，直接跳出
    if(pMP->mnId==this->mnId)
        return;

    int nvisible, nfound;
    map<KeyFrame*,size_t> obs; // 
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        obs=mObservations;
        mObservations.clear();
        mbBad=true;
        nvisible = mnVisible;
        nfound = mnFound;
        mpReplaced = pMP;
    }

    // For all observations of this MapPoint
    for(map<KeyFrame*,size_t>::iterator mit=obs.begin(), mend=obs.end(); mit!=mend; mit++)
    {
        // Replace measurement in keyframe
        // 
        KeyFrame* pKF = mit->first; // keyframe

        // pMP is not in the pKF
        if(!pMP->IsInKeyFrame(pKF))
        {
            // add new observation
            // mit->first:  KeyFrame
            // mit->second: idx of MapPoint
            // find the old idx of MapPoint in obs, replace with pMP (new one), idx is fixed
            pKF->ReplaceMapPointMatch(mit->second, pMP);
            // add the new observation for pMP
            pMP->AddObservation(pKF, mit->second);
        }
        else
        {   
            // delete old observations
            pKF->EraseMapPointMatch(mit->second);
        }
    }
    pMP->IncreaseFound(nfound);
    pMP->IncreaseVisible(nvisible);
    pMP->ComputeDistinctiveDescriptors();

    // replace old mappoint
    mpMap->EraseMapPoint(this);
}


bool MapPoint::isBad()
{
    unique_lock<mutex> lock(mMutexFeatures);
    unique_lock<mutex> lock2(mMutexPos);
    return mbBad;
}

void MapPoint::IncreaseVisible(int n)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mnVisible+=n;
}

void MapPoint::IncreaseFound(int n)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mnFound+=n;
}

float MapPoint::GetFoundRatio()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return static_cast<float>(mnFound)/mnVisible;
}


/**
 * For one MapPoint
 * 由于一个 MapPoint 会被 许多相机 观测到，
 * 因此在插入 关键帧 后，需要判断是否更新 当前点(MapPoint) 的最适合的 描述子,
 * 
 * 最好的描述子 与 其他描述子 应该具有 最小的平均距离 ，因此先获得当前点的 所有 描述子 ，
 * 然后计算描述子之间的 两两距离 ，对所有距离取平均 ， 最后找离这个 中值距离 最近 的描述子。
 * 
 * choose the best one among all candidate based on the comparision between each pair of descriptors
 */
void MapPoint::ComputeDistinctiveDescriptors()
{
    // Retrieve all observed descriptors
    vector<cv::Mat> vDescriptors;

    map<KeyFrame*,size_t> observations;

    {
        unique_lock<mutex> lock1(mMutexFeatures);
        // 如果地图点标记为不好，直接返回
        if(mbBad)
            return;
        observations=mObservations;
    }

    // 如果观测为空，则返回
    if(observations.empty())
        return;

    // for one MapPoint
    // 保留的 描述子数(vDescriptors) 最多和 观测数(observations) 一致
    vDescriptors.reserve(observations.size());

    // a mappoint can be observed by many keyframes, so there are many observations
    // one observation <-> one keyframe
    // map<KeyFrame, idx of MapPoint> observations;
    // For different KF -> observations[KF]
    for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;

        if(!pKF->isBad())
            // for each mappoint, every frame provides a descriptor
            // 针对每帧的对应的都提取其 描述子
            // cv::Mat mDescriptors
            // 每个特征点描述子占一行，建立一个指针指向iL特征点对应的描述子
            // vDescriptors storages the descriptors of one MapPoints in each KeyFrame
            // mit->second:                         idx of MapPoint in this KF
            // pKF->mDescriptors.row(mit->second):  descriptor of MapPoint in this KF
            vDescriptors.push_back(pKF->mDescriptors.row(mit->second));
    }

    if(vDescriptors.empty())
        return;

    // Compute distances between them
    const size_t N = vDescriptors.size();

    float Distances[N][N];
    for(size_t i=0;i<N;i++)
    {
        Distances[i][i]=0;
        for(size_t j=i+1;j<N;j++)
        {
            int distij = ORBmatcher::DescriptorDistance(vDescriptors[i],vDescriptors[j]);
            Distances[i][j]=distij;
            Distances[j][i]=distij;
        }
    }

    // Take the descriptor with least median distance to the rest
    // 选择距离 其他描述子中值距离 最小 的 描述子 作为 地图点 的 描述子 ，基本上类似于取了个均值
    int BestMedian = INT_MAX;
    int BestIdx = 0;
    for(size_t i=0;i<N;i++)
    {
        vector<int> vDists(Distances[i],Distances[i]+N);
        sort(vDists.begin(),vDists.end());
        int median = vDists[0.5*(N-1)];

        // choose the minimum
        if(median<BestMedian) 
        {
            BestMedian = median;
            BestIdx = i;
        }
    }

    {
        unique_lock<mutex> lock(mMutexFeatures);
        // choose the best descriptor for a MapPoint
        mDescriptor = vDescriptors[BestIdx].clone();
    }
}

cv::Mat MapPoint::GetDescriptor()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mDescriptor.clone();
}

int MapPoint::GetIndexInKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexFeatures);
    if(mObservations.count(pKF))
        return mObservations[pKF]; // idx
    else
        return -1;
}

bool MapPoint::IsInKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexFeatures);
    return (mObservations.count(pKF));
}


/**
 * @brief
 * For one MapPoint
 * 由于图像提取 描述子 是使用 金字塔 分层提取，所以计算法向量和深度可以知道
 * 该 MapPoint 在对应的关键帧的金字塔哪一层可以提取到
 * 明确了目的，下一步就是方法问题，所谓的 法向量 ，就是说 相机光心 指向地图点的方向，
 * 计算这个方向方法很简单，只需要用 地图点的三维坐标 减去 相机光心的三维坐标 就可以
 */
void MapPoint::UpdateNormalAndDepth()
{
    map<KeyFrame*,size_t> observations;                     // observations[KF] = idx of MapPoint
    KeyFrame* pRefKF;
    cv::Mat Pos;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        if(mbBad)
            return;
        observations=mObservations;
        pRefKF=mpRefKF;
        Pos = mWorldPos.clone();
    }

    if(observations.empty())
        return;

    cv::Mat normal = cv::Mat::zeros(3,1,CV_32F);
    int n=0;

    // one mappoint can be observed by many keyframes
    // So, for each observation/keyframe, calculate the normal
    for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;                         // KeyFrame
        cv::Mat Owi = pKF->GetCameraCenter();               // camera center
        /** 
         * normal vector, observation direction
         * camera center points to mappoint
         * 观测点坐标 减去 关键帧中 相机光心的坐标 就是 观测方向
         * 也就是说相机光心指向地图点
         */
        cv::Mat normali = mWorldPos - Owi;

        // 对其进行 归一化 后 相加
        normal = normal + normali/cv::norm(normali);
        n++;
    }

    cv::Mat PC = Pos - pRefKF->GetCameraCenter();
    const float dist = cv::norm(PC);
    const int level = pRefKF->mvKeysUn[observations[pRefKF]].octave; // pRefKF->mvKeysUn[observations[pRefKF]]: MapPoint<->KeyPoint
    const float levelScaleFactor =  pRefKF->mvScaleFactors[level];
    const int nLevels = pRefKF->mnScaleLevels;

    /**
     * 深度范围 [mfMinDistance, mfMaxDistance]
     * 地图点到参考帧（只有一帧）相机中心距离 乘上 参考帧中描述子获取金字塔放大尺度，得到最大距离 mfMaxDistance
     * 最大距离 除以 整个金字塔最高层 的 放大尺度 得到 最小距离 mfMinDistance
     * 通常说来，距离较近的地图点，将在金字塔较高的地方提出，
     * 距离较远的地图点，在金字塔层数较低的地方提取出（金字塔层数越低，分辨率越高，才能识别出远点）
     * 因此，通过地图点的信息（主要对应描述子），我们可以获得该地图点对应的金字塔层级
     * 从而预测该地图点在什么范围内能够被观测到
     */
    {
        unique_lock<mutex> lock3(mMutexPos);
        mfMaxDistance = dist*levelScaleFactor;
        mfMinDistance = mfMaxDistance/pRefKF->mvScaleFactors[nLevels-1];
        mNormalVector = normal/n;                       // updated mean normal vector
    }
}

float MapPoint::GetMinDistanceInvariance()
{
    unique_lock<mutex> lock(mMutexPos);
    return 0.8f*mfMinDistance;
}

float MapPoint::GetMaxDistanceInvariance()
{
    unique_lock<mutex> lock(mMutexPos);
    return 1.2f*mfMaxDistance;
}


/**
 * 该函数的作用是 预测 特征点 在 金字塔 哪一层 可以找到
 * 
 * 注意金字塔ScaleFactor和距离的关系：
 * 当特征点对应 ScaleFactor 为1.2的意思是：图片分辨率下降 1.2 倍后，可以提取出该特征点
 * (分辨率更高的时候，肯定也可以提出，这里取 金字塔 中能够提取出该特征点 最高层级 作为该特征点的层级)，
 * 同时，由当前特征点的距离，推测所在的层级。
 * 
 * 
 *              ________
 * Nearer      /________\           level: n-1  --> dmin
 *            /__________\                                  d/dmin = 1.2^(n-1-m)
 *           /____________\         level: m    --> d
 *          /______________\
 *         /________________\                               dmax/d = 1.2^m
 * Farther/__________________\      level: 0    --> dmax
 * 
 *             log(dmax/d)
 * m = ceil (---------------)
 *              log(1.2)
 * 
 * @param currentDist   当前距离
 * @param pKF           关键帧
 */
int MapPoint::PredictScale(const float &currentDist, KeyFrame* pKF)
{
    float ratio;
    {
        unique_lock<mutex> lock(mMutexPos);
        ratio = mfMaxDistance/currentDist;
    }

    int nScale = ceil(log(ratio)/pKF->mfLogScaleFactor);
    if(nScale<0)
        nScale = 0;
    else if(nScale>=pKF->mnScaleLevels)
        nScale = pKF->mnScaleLevels-1;

    return nScale;
}


int MapPoint::PredictScale(const float &currentDist, Frame* pF)
{
    float ratio;
    {
        unique_lock<mutex> lock(mMutexPos);
        ratio = mfMaxDistance/currentDist;
    }

    int nScale = ceil(log(ratio)/pF->mfLogScaleFactor);
    if(nScale<0)
        nScale = 0;
    else if(nScale>=pF->mnScaleLevels)
        nScale = pF->mnScaleLevels-1;

    return nScale;
}



} //namespace ORB_SLAM
