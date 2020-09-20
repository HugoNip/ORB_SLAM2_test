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

#include "Frame.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include <thread>

namespace ORB_SLAM2
{

long unsigned int Frame::nNextId=0;
bool Frame::mbInitialComputations=true;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

Frame::Frame()
{}

//Copy Constructor
Frame::Frame(const Frame &frame)
    :mpORBvocabulary(frame.mpORBvocabulary), 
     mpORBextractorLeft(frame.mpORBextractorLeft), 
     mpORBextractorRight(frame.mpORBextractorRight),
     mTimeStamp(frame.mTimeStamp), 
     mK(frame.mK.clone()), 
     mDistCoef(frame.mDistCoef.clone()),
     mbf(frame.mbf), 
     mb(frame.mb), 
     mThDepth(frame.mThDepth), 
     N(frame.N), 
     mvKeys(frame.mvKeys),
     mvKeysRight(frame.mvKeysRight), 
     mvKeysUn(frame.mvKeysUn), 
     mvuRight(frame.mvuRight),
     mvDepth(frame.mvDepth), 
     mBowVec(frame.mBowVec), 
     mFeatVec(frame.mFeatVec),
     mDescriptors(frame.mDescriptors.clone()), 
     mDescriptorsRight(frame.mDescriptorsRight.clone()),
     mvpMapPoints(frame.mvpMapPoints), 
     mvbOutlier(frame.mvbOutlier), 
     mnId(frame.mnId),
     mpReferenceKF(frame.mpReferenceKF), 
     mnScaleLevels(frame.mnScaleLevels),
     mfScaleFactor(frame.mfScaleFactor), 
     mfLogScaleFactor(frame.mfLogScaleFactor),
     mvScaleFactors(frame.mvScaleFactors), 
     mvInvScaleFactors(frame.mvInvScaleFactors),
     mvLevelSigma2(frame.mvLevelSigma2), 
     mvInvLevelSigma2(frame.mvInvLevelSigma2) {
    for(int i=0;i<FRAME_GRID_COLS;i++)
        for(int j=0; j<FRAME_GRID_ROWS; j++)
            mGrid[i][j]=frame.mGrid[i][j];

    if(!frame.mTcw.empty())
        SetPose(frame.mTcw);
}

// stereo
Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, 
             const double &timeStamp, 
             ORBextractor* extractorLeft, 
             ORBextractor* extractorRight, 
             ORBVocabulary* voc, 
             cv::Mat &K, // camera intrinsics
             cv::Mat &distCoef, 
             const float &bf, 
             const float &thDepth) // threshold of depth
    :mpORBvocabulary(voc), 
     mpORBextractorLeft(extractorLeft),
     mpORBextractorRight(extractorRight), 
     mTimeStamp(timeStamp), 
     mK(K.clone()),
     mDistCoef(distCoef.clone()), 
     mbf(bf), 
     mThDepth(thDepth),
     mpReferenceKF(static_cast<KeyFrame*>(NULL)) {
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    thread threadLeft(&Frame::ExtractORB,this,0,imLeft);
    thread threadRight(&Frame::ExtractORB,this,1,imRight);
    threadLeft.join();
    threadRight.join();

    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    // compute matches, 
    // store depth, left + right coordinates of keypoints
    ComputeStereoMatches();

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));    
    mvbOutlier = vector<bool>(N,false);


    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imLeft);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();
}

// RGBD
Frame::Frame(const cv::Mat &imGray, 
             const cv::Mat &imDepth, 
             const double &timeStamp, 
             ORBextractor* extractor,
             ORBVocabulary* voc, 
             cv::Mat &K, 
             cv::Mat &distCoef, 
             const float &bf, 
             const float &thDepth)
    :mpORBvocabulary(voc),
     mpORBextractorLeft(extractor),
     mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), 
     mK(K.clone()),
     mDistCoef(distCoef.clone()), 
     mbf(bf), 
     mThDepth(thDepth) {
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();    
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    ExtractORB(0,imGray);

    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    ComputeStereoFromRGBD(imDepth);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();
}

// monocular
Frame::Frame(const cv::Mat &imGray, 
             const double &timeStamp, 
             ORBextractor* extractor,
             ORBVocabulary* voc, 
             cv::Mat &K, 
             cv::Mat &distCoef, 
             const float &bf, 
             const float &thDepth)
    :mpORBvocabulary(voc),
     mpORBextractorLeft(extractor),
     mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), 
     mK(K.clone()),
     mDistCoef(distCoef.clone()), 
     mbf(bf), 
     mThDepth(thDepth) {
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    ExtractORB(0,imGray);

    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    // Set no stereo information
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();
}

void Frame::AssignFeaturesToGrid()
{
    int nReserve = 0.5f*N/(FRAME_GRID_COLS*FRAME_GRID_ROWS);
    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++)
            mGrid[i][j].reserve(nReserve);

    for(int i=0;i<N;i++)
    {
        const cv::KeyPoint &kp = mvKeysUn[i];

        int nGridPosX, nGridPosY;
        if(PosInGrid(kp,nGridPosX,nGridPosY))
            mGrid[nGridPosX][nGridPosY].push_back(i);
    }
}

void Frame::ExtractORB(int flag, const cv::Mat &im)
{
    if(flag==0) // left
        (*mpORBextractorLeft)(im,cv::Mat(),mvKeys,mDescriptors);
    else // right
        (*mpORBextractorRight)(im,cv::Mat(),mvKeysRight,mDescriptorsRight);
}


/**
 * set initial pose for current frame
 */
void Frame::SetPose(cv::Mat Tcw)
{
    mTcw = Tcw.clone();
    UpdatePoseMatrices();
}

void Frame::UpdatePoseMatrices()
{ 
    mRcw = mTcw.rowRange(0,3).colRange(0,3);
    mRwc = mRcw.t(); // cw->wc
    mtcw = mTcw.rowRange(0,3).col(3);
    mOw = -mRcw.t()*mtcw; // camera center
}

bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit)
{
    /**
     * 1. coordinate (u, v) in the image
     * 2. distance between camera center O and mappoint
     * 3. view angle
     * 4. scale
     */
    pMP->mbTrackInView = false;

    // 3D in absolute coordinates
    cv::Mat P = pMP->GetWorldPos(); 

    // 3D in camera coordinates
    const cv::Mat Pc = mRcw*P+mtcw; // P_c = R_cw * P_w + t_cw
    const float &PcX = Pc.at<float>(0);
    const float &PcY= Pc.at<float>(1);
    const float &PcZ = Pc.at<float>(2);

    // Check positive depth
    if(PcZ<0.0f)
        return false;

    // Project in image (u, v) and check it is not outside
    const float invz = 1.0f/PcZ;
    const float u=fx*PcX*invz+cx;
    const float v=fy*PcY*invz+cy;

    // Check whether or not (u, v) is in the image
    if(u<mnMinX || u>mnMaxX)
        return false;
    if(v<mnMinY || v>mnMaxY)
        return false;

    // Check distance is in the scale invariance region of the MapPoint
    // 计算MapPoint到相机中心的距离, 并判断是否在尺度变化的距离内
    // 每一个地图点都是对应于若干尺度的金字塔提取出来的，具有一定的有效深度
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    // 世界坐标系下，相机到3D点P的向量, 向量方向由相机指向3D点P
    const cv::Mat PO = P-mOw; // vector from camera center to mappoint
    const float dist = cv::norm(PO); // absolute L2 norm

    if(dist<minDistance || dist>maxDistance)
        return false;

    // Check viewing angle
    // 计算当前视角和平均视角夹角的余弦值, 若小于cos(60), 即夹角大于60度则返回
    // 每一个地图都有其平均视角，是从能够观测到地图点的帧位姿中计算出
    cv::Mat Pn = pMP->GetNormal(); // normal vector

    const float viewCos = PO.dot(Pn)/dist;

    if(viewCos<viewingCosLimit)
        return false;

    // Predict scale in the image
    // 根据深度预测尺度（对应特征点在一层）
    const int nPredictedLevel = pMP->PredictScale(dist,this);

    // Data used by the tracking
    // if mappoint is in the view, this mappoint will be used by tracking
    pMP->mbTrackInView = true;
    pMP->mTrackProjX = u;
    pMP->mTrackProjXR = u - mbf*invz; // x coordinate in the right image, has disparity
    pMP->mTrackProjY = v;
    pMP->mnTrackScaleLevel= nPredictedLevel;
    pMP->mTrackViewCos = viewCos;

    return true;
}


vector<size_t> Frame::GetFeaturesInArea(
    const float &x, 
    const float &y, 
    const float &r, 
    const int minLevel, 
    const int maxLevel) const {
    
    /**
     * void reserve(size_type n)
     * 
     * std::vector class provides a useful function reserve which helps user 
     * specify the minimum size of the vector. It indicates that the vector 
     * is created such that it can store at least the number of the specified 
     * elements without having to reallocate memory.
     * 
     * Requests that vector is large enough to store n elements in the least.
     */
    vector<size_t> vIndices;
    vIndices.reserve(N);

    /**
     * 接下来计算方形的四边在哪，在 mGrid 中的行数和列数
     * nMinCellX 是方形(Cell)左边在 mGrid 中的列数，
     * 如果它比 mGrid 的列数大，说明方形内肯定没有特征点，于是返回
     */
    const int nMinCellX = max(0, (int)floor((x-mnMinX-r)*mfGridElementWidthInv));
    if(nMinCellX>=FRAME_GRID_COLS)
        return vIndices;

    const int nMaxCellX = min((int)FRAME_GRID_COLS-1,
                              (int)ceil((x-mnMinX+r)*mfGridElementWidthInv));
    if(nMaxCellX<0)
        return vIndices;

    const int nMinCellY = max(0, (int)floor((y-mnMinY-r)*mfGridElementHeightInv));
    if(nMinCellY>=FRAME_GRID_ROWS)
        return vIndices;

    const int nMaxCellY = min((int)FRAME_GRID_ROWS-1,
                              (int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;

    const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

    // crop keypoints in the area
    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            // Keypoints are assigned to cells in a grid to reduce matching complexity 
            // when projecting MapPoints.
            const vector<size_t> vCell = mGrid[ix][iy]; // cell in a grid
            if(vCell.empty())
                continue;

            // keypoints in cells
            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                // mvKeysUn: Vector of keypoints (original for visualization) and undistorted
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                if(bCheckLevels)
                {
                    if(kpUn.octave<minLevel)
                        continue;
                    if(maxLevel>=0)
                        if(kpUn.octave>maxLevel)
                            continue;
                }

                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                // put all feature points in the area into a vector
                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
    posX = round((kp.pt.x-mnMinX)*mfGridElementWidthInv);
    posY = round((kp.pt.y-mnMinY)*mfGridElementHeightInv);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
        return false;

    return true;
}


/**
 * @brief Bag of Words Representation
 *
 * 计算词包mBowVec和mFeatVec，其中mFeatVec记录了属于第i个node（在第4层）的ni个描述子
 * 
 * 如果没有传入已有的词袋数据，则就用当前的描述子重新计算生成词袋数据
 * 
 * @see CreateInitialMapMonocular() TrackReferenceKeyFrame() Relocalization()
 */
void Frame::ComputeBoW()
{
    if(mBowVec.empty())
    {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);     // 当前的描述子
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
}


void Frame::UndistortKeyPoints()
{
    if(mDistCoef.at<float>(0)==0.0)
    {
        mvKeysUn=mvKeys;
        return;
    }

    // Fill matrix with points
    cv::Mat mat(N,2,CV_32F);
    for(int i=0; i<N; i++)
    {
        mat.at<float>(i,0)=mvKeys[i].pt.x;
        mat.at<float>(i,1)=mvKeys[i].pt.y;
    }

    // Undistort points
    mat=mat.reshape(2);
    cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
    mat=mat.reshape(1);

    // Fill undistorted keypoint vector
    mvKeysUn.resize(N);
    for(int i=0; i<N; i++)
    {
        cv::KeyPoint kp = mvKeys[i];
        kp.pt.x=mat.at<float>(i,0);
        kp.pt.y=mat.at<float>(i,1);
        mvKeysUn[i]=kp;
    }
}

void Frame::ComputeImageBounds(const cv::Mat &imLeft)
{
    if(mDistCoef.at<float>(0)!=0.0)
    {
        cv::Mat mat(4,2,CV_32F);
        mat.at<float>(0,0)=0.0; 
        mat.at<float>(0,1)=0.0;

        mat.at<float>(1,0)=imLeft.cols; 
        mat.at<float>(1,1)=0.0;

        mat.at<float>(2,0)=0.0; 
        mat.at<float>(2,1)=imLeft.rows;
        
        mat.at<float>(3,0)=imLeft.cols; 
        mat.at<float>(3,1)=imLeft.rows;

        // Undistort corners
        mat=mat.reshape(2);
        cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
        mat=mat.reshape(1);

        mnMinX = min(mat.at<float>(0,0),mat.at<float>(2,0));
        mnMaxX = max(mat.at<float>(1,0),mat.at<float>(3,0));
        mnMinY = min(mat.at<float>(0,1),mat.at<float>(1,1));
        mnMaxY = max(mat.at<float>(2,1),mat.at<float>(3,1));

    }
    else
    {
        mnMinX = 0.0f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.0f;
        mnMaxY = imLeft.rows;
    }
}


/**
 * 其作用是为左图的每一个特征点在右图中找到匹配点，
 * 根据基线(有冗余范围)上描述子距离找到匹配，
 * 再进行SAD精确定位，最后对所有SAD的值进行排序, 剔除SAD值较大的匹配对，
 * 然后利用抛物线拟合得到亚像素精度的匹配，匹配成功后会更新 mvuRight 和 mvDepth
 */
void Frame::ComputeStereoMatches()
{
    mvuRight = vector<float>(N,-1.0f);
    mvDepth = vector<float>(N,-1.0f);

    const int thOrbDist = (ORBmatcher::TH_HIGH+ORBmatcher::TH_LOW)/2;

    const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;

    // Assign keypoints to row table
    // 步骤1：建立特征点搜索范围对应表，一个特征点在一个带状区域内搜索匹配特征点
    // 匹配搜索的时候，不仅仅是在一条横线上搜索，而是在一条横向搜索带上搜索,
    // 简而言之，原本每个特征点的纵坐标为1，这里把特征点体积放大，纵坐标占好几行
    // 例如左目图像某个特征点的纵坐标为20，那么在右侧图像上搜索时是在纵坐标为18到22这条带上搜索，
    // 搜索带宽度为正负2，搜索带的宽度和特征点所在金字塔层数有关
    // 简单来说，如果纵坐标是20，特征点在图像第20行，那么认为18 19 20 21 22行都有这个特征点
    // vRowIndices[18]、vRowIndices[19]、vRowIndices[20]、vRowIndices[21]、
    // vRowIndices[22]都有这个特征点编号

    vector<vector<size_t> > vRowIndices(nRows,vector<size_t>());

    for(int i=0; i<nRows; i++)
        vRowIndices[i].reserve(200);

    const int Nr = mvKeysRight.size();

    for(int iR=0; iR<Nr; iR++)
    {
        const cv::KeyPoint &kp = mvKeysRight[iR];
        const float &kpY = kp.pt.y;
        // 计算匹配搜索的纵向宽度，尺度越大（层数越高，距离越近），搜索范围越大
        // 如果特征点在金字塔第一层，则搜索范围为:正负2
        // 尺度越大其位置不确定性越高，所以其搜索半径越大
        const float r = 2.0f*mvScaleFactors[mvKeysRight[iR].octave];
        const int maxr = ceil(kpY+r);
        const int minr = floor(kpY-r);

        for(int yi=minr;yi<=maxr;yi++)
            vRowIndices[yi].push_back(iR);
    }

    // Set limits for search
    const float minZ = mb; // NOTE bug mb没有初始化，mb的赋值在构造函数中放在ComputeStereoMatches函数的后面
    const float minD = 0; // 最小视差, 设置为0即可
    const float maxD = mbf/minZ; // 最大视差, 对应最小深度 mbf/minZ = mbf/mb = mbf/(mbf/fx) = fx

    // For each left keypoint search a match in the right image
    vector<pair<int, int> > vDistIdx;
    vDistIdx.reserve(N);

    // 步骤2：对左目相机每个特征点，通过描述子在右目带状搜索区域找到匹配点, 再通过SAD做亚像素匹配
    // 注意：1)这里是校正前的mvKeys，而不是校正后的mvKeysUn
    //      2)KeyFrame::UnprojectStereo和Frame::UnprojectStereo函数中不一致
    for(int iL=0; iL<N; iL++)
    {
        const cv::KeyPoint &kpL = mvKeys[iL];
        const int &levelL = kpL.octave;
        const float &vL = kpL.pt.y;
        const float &uL = kpL.pt.x;

        // 可能的匹配点
        const vector<size_t> &vCandidates = vRowIndices[vL];

        if(vCandidates.empty())
            continue;

        const float minU = uL-maxD; // 最小匹配范围
        const float maxU = uL-minD; // 最大匹配范围

        if(maxU<0)
            continue;

        int bestDist = ORBmatcher::TH_HIGH;
        size_t bestIdxR = 0;

        // 每个特征点描述子占一行，建立一个指针指向iL特征点对应的描述子
        const cv::Mat &dL = mDescriptors.row(iL);

        // Compare descriptor to right keypoints
        // 步骤2.1：遍历右目所有可能的匹配点，找出最佳匹配点（描述子距离最小）
        for(size_t iC=0; iC<vCandidates.size(); iC++)
        {
            const size_t iR = vCandidates[iC];
            const cv::KeyPoint &kpR = mvKeysRight[iR];
            
            // 仅对近邻尺度的特征点进行匹配
            if(kpR.octave<levelL-1 || kpR.octave>levelL+1)
                continue;

            const float &uR = kpR.pt.x;
            // 此处找出的 bestIdxR 就是最匹配的特征点，bestDist 是该特征点对应的描述向量距离
            if(uR>=minU && uR<=maxU)
            {
                const cv::Mat &dR = mDescriptorsRight.row(iR);
                const int dist = ORBmatcher::DescriptorDistance(dL,dR);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdxR = iR;
                }
            }
        }

        // Subpixel match by correlation
        // 步骤2.2：通过SAD匹配提高像素匹配修正量bestincR
        if(bestDist<thOrbDist)
        {
            // coordinates in image pyramid at keypoint scale
            // kpL.pt.x对应金字塔最底层坐标，将最佳匹配的 特征点对 尺度变换到尺度对应层 
            // (scaleduL, scaledvL) (scaleduR0, )
            const float uR0 = mvKeysRight[bestIdxR].pt.x;
            const float scaleFactor = mvInvScaleFactors[kpL.octave];
            const float scaleduL = round(kpL.pt.x*scaleFactor);
            const float scaledvL = round(kpL.pt.y*scaleFactor);
            const float scaleduR0 = round(uR0*scaleFactor); // best matching point

            // sliding window search
            const int w = 5; // 滑动窗口的大小11*11 注意该窗口取自resize后的图像
            cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduL-w,scaleduL+w+1);
            IL.convertTo(IL,CV_32F);
            IL = IL - IL.at<float>(w,w) *cv::Mat::ones(IL.rows,IL.cols,CV_32F);

            int bestDist = INT_MAX;
            int bestincR = 0;
            const int L = 5;
            vector<float> vDists;
            vDists.resize(2*L+1);

            // 滑动窗口的滑动范围为（-L, L）,提前判断滑动窗口滑动过程中是否会越界
            const float iniu = scaleduR0+L-w;
            const float endu = scaleduR0+L+w+1;
            if(iniu<0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
                continue;

            for(int incR=-L; incR<=+L; incR++)
            {
                // 横向滑动窗口
                cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduR0+incR-w,scaleduR0+incR+w+1);
                IR.convertTo(IR,CV_32F);
                //窗口中的每个元素减去正中心的那个元素，简单归一化，减小光照强度影响
                IR = IR - IR.at<float>(w,w) *cv::Mat::ones(IR.rows,IR.cols,CV_32F);

                float dist = cv::norm(IL,IR,cv::NORM_L1); // 一范数，计算差的绝对值
                if(dist<bestDist)
                {
                    bestDist =  dist; // SAD匹配目前最小匹配偏差
                    bestincR = incR;  // SAD匹配目前最佳的修正量
                }

                vDists[L+incR] = dist;
            }

            // 整个滑动窗口过程中，SAD最小值不是以抛物线形式出现，SAD匹配失败，同时放弃求该特征点的深度
            if(bestincR==-L || bestincR==L)
                continue;

            // Sub-pixel match (Parabola fitting)
            // 步骤2.3：做抛物线拟合找谷底得到亚像素匹配deltaR
            // (bestincR,dist) (bestincR-1,dist) (bestincR+1,dist)三个点拟合出抛物线
            // bestincR+deltaR 就是抛物线 谷底 的位置，相对SAD匹配出的最小值 bestincR 的修正量为 deltaR
            const float dist1 = vDists[L+bestincR-1];
            const float dist2 = vDists[L+bestincR];
            const float dist3 = vDists[L+bestincR+1];

            const float deltaR = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

            // 抛物线拟合得到的修正量不能超过一个像素，否则放弃求该特征点的深度
            if(deltaR<-1 || deltaR>1)
                continue;

            // Re-scaled coordinate
            // 通过描述子匹配得到匹配点位置为 scaleduR0
            // 通过SAD匹配找到修正量 bestincR
            // 通过抛物线拟合找到亚像素修正量 deltaR
            float bestuR = mvScaleFactors[kpL.octave]*
                         ((float)scaleduR0+(float)bestincR+deltaR);

            // 这里是disparity，根据它算出depth
            float disparity = (uL-bestuR);

            // 最后判断视差是否在范围内
            if(disparity>=minD && disparity<maxD)
            {
                if(disparity<=0)
                {
                    disparity=0.01;
                    bestuR = uL-0.01;
                }
                // depth 是在这里计算的
                // depth=baseline*fx/disparity
                mvDepth[iL]=mbf/disparity; // 深度
                mvuRight[iL] = bestuR; // 匹配对在右图的横坐标
                vDistIdx.push_back(pair<int,int>(bestDist,iL)); // 该特征点SAD匹配最小匹配偏差
            }
        }
    }

    // 步骤3：剔除SAD匹配偏差较大的匹配特征点
    // 前面SAD匹配只判断滑动窗口中是否有局部最小值，
    // 这里通过对比剔除SAD匹配偏差比较大的特征点的深度
    sort(vDistIdx.begin(),vDistIdx.end()); // 根据所有匹配对的SAD偏差进行排序, 距离由小到大
    const float median = vDistIdx[vDistIdx.size()/2].first; 
    const float thDist = 1.5f*1.4f*median; // 计算自适应距离, 大于此距离的匹配对将剔除

    for(int i=vDistIdx.size()-1;i>=0;i--)
    {
        if(vDistIdx[i].first<thDist)
            break;
        else
        {
            mvuRight[vDistIdx[i].second]=-1;
            mvDepth[vDistIdx[i].second]=-1;
        }
    }
}

// 根据像素坐标获取深度信息，如果深度存在则保存下来，这里还计算了假想右图的对应特征点的横坐标
void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth)
{
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    for(int i=0; i<N; i++)
    {
        const cv::KeyPoint &kp = mvKeys[i]; // distorted image
        const cv::KeyPoint &kpU = mvKeysUn[i]; // undistorted image

        const float &v = kp.pt.y;
        const float &u = kp.pt.x;

        const float d = imDepth.at<float>(v,u);

        if(d>0)
        {
            mvDepth[i] = d;
            mvuRight[i] = kpU.pt.x-mbf/d; // 假想右图的对应特征点的横坐标
        }
    }
}

// 其作用是将特征点坐标反投影到3D地图点（世界坐标），
// 在已知深度的情况下，则可确定二维像素点对应的尺度，最后获得3D中点坐标
cv::Mat Frame::UnprojectStereo(const int &i)
{
    const float z = mvDepth[i];
    if(z>0)
    {
        const float u = mvKeysUn[i].pt.x;
        const float v = mvKeysUn[i].pt.y;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);
        return mRwc*x3Dc+mOw; // transformation
    }
    else
        return cv::Mat();
}

} //namespace ORB_SLAM
