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


#ifndef TRACKING_H
#define TRACKING_H

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include"Viewer.h"
#include"FrameDrawer.h"
#include"Map.h"
#include"LocalMapping.h"
#include"LoopClosing.h"
#include"Frame.h"
#include "ORBVocabulary.h"
#include"KeyFrameDatabase.h"
#include"ORBextractor.h"
#include "Initializer.h"
#include "MapDrawer.h"
#include "System.h"

#include <mutex>

namespace ORB_SLAM2
{

class Viewer;
class FrameDrawer;
class Map;
class LocalMapping;
class LoopClosing;
class System;

/**
 * Define the Tracker. 
 * It receives a frame and computes the associated camera pose.
 * It also decides when to insert a new keyframe, create some new MapPoints and
 * performs **relocalization** if tracking fails.
 * --------------------------
 * Tracking* mpTracker;
 * 
 * 系统将图片交给Tracking::GrabImageMonocular()后，先将图片转化为灰度图，然后使用图片构建了一个Frame。
 * 注意系统在初始化的时候使用了不同的ORBextractor来构建Frame，是应为在初始化阶段的帧需要跟多的特征点
 * 
 * Tracking执行了4个任务：
 * 1. 单目初始化
 * 2. 通过 上一帧 获得 初始位姿估计或者重定位。也就是求出当前帧在世界坐标系下的**位姿T2**
 * 3. 跟踪局部地图（TrackLocalMap()） 求得**位姿估计T3**。这个步骤回答当前帧的特征点和map中的哪些mappoint**匹配**。
 * 4. 判断是否需要给localmapping插入关键帧
 */
class Tracking
{  

public:
    Tracking(System* pSys, ORBVocabulary* pVoc, FrameDrawer* pFrameDrawer, MapDrawer* pMapDrawer, Map* pMap,
             KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor);

    // Preprocess the input and call Track(). Extract features and performs stereo matching.
    cv::Mat GrabImageStereo(const cv::Mat &imRectLeft,const cv::Mat &imRectRight, const double &timestamp);
    cv::Mat GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp);
    cv::Mat GrabImageMonocular(const cv::Mat &im, const double &timestamp);

    void SetLocalMapper(LocalMapping* pLocalMapper);
    void SetLoopClosing(LoopClosing* pLoopClosing);
    void SetViewer(Viewer* pViewer);

    // Load new settings
    // The focal length should be similar or scale prediction will fail when projecting points
    // TODO: Modify MapPoint::PredictScale to take into account focal length
    void ChangeCalibration(const string &strSettingPath);

    // Use this function if you have **deactivated local mapping** and you only want to localize the camera.
    void InformOnlyTracking(const bool &flag);


public:

    // Tracking states
    enum eTrackingState{
        SYSTEM_NOT_READY=-1,
        NO_IMAGES_YET=0,
        NOT_INITIALIZED=1,
        OK=2,
        LOST=3
    };

    eTrackingState mState;
    eTrackingState mLastProcessedState;

    // Input sensor
    int mSensor;

    // Current Frame
    Frame mCurrentFrame;
    cv::Mat mImGray;

    // Initialization Variables (Monocular)
    std::vector<int> mvIniLastMatches;

    // 初始化时 得到的 特征点匹配，大小是 mInitialFrame 的特征点数量，其值是 当前帧特征点 序号(idx)
    std::vector<int> mvIniMatches;

    // mInitialFrame 中待匹配的特征点的 像素位置
    std::vector<cv::Point2f> mvbPrevMatched;    // [u, v]       keypoints
    
    // 初始化时 三角化投影成功的匹配点对应的 3d点
    std::vector<cv::Point3f> mvIniP3D;          // [X, Y, Z]    MapPoints
    // 初始化的 第一帧，初始化需要两帧,世界坐标系就是这帧的坐标系
    Frame mInitialFrame;

    // Lists used to recover the full camera trajectory at the end of the execution.
    // Basically we store the **reference keyframe** for each frame and its **relative transformation**
    list<cv::Mat> mlRelativeFramePoses; // Relative Frame Poses
    list<KeyFrame*> mlpReferences;      // reference keyframe
    list<double> mlFrameTimes;
    list<bool> mlbLost;

    // True if local mapping is deactivated and we are performing only localization
    bool mbOnlyTracking;

    void Reset();

protected:

    // Main tracking function. It is independent of the input sensor.
    void Track();

    // Map initialization for stereo and RGB-D
    void StereoInitialization();

    // Map initialization for monocular
    /**
    * @brief 单目 的地图初始化
    *
    * 并行地计算基础矩阵和单应性矩阵，选取其中一个模型，恢复出最开始两帧之间的相对姿态以及点云
    * 得到初始两帧的匹配、相对运动、初始 MapPoints
    */
    void MonocularInitialization();

    /**
     * 单目模式下 初始化后，开始建图
     * 将 mInitialFrame 和 mCurrentFrame 都设置为关键帧
     * 新建 mappoint
     * 更新 共视关系
     */
    void CreateInitialMapMonocular();

    void CheckReplacedInLastFrame();

    /**
    * 将上一帧的位姿 作为 当前帧 mCurrentFrame 的 初始位姿；
    * 匹配参考帧关键帧 中有对应mappoint的特征点与当前帧特征点，通过dbow加速 **匹配**
    * 以上一帧的位姿态为 初始值，优化3D点重投影误差，**得到(mCurrentFrame)更精确的位姿** 以及剔除错误的特征点匹配；
    * 
    * @return 如果匹配数大于10，返回true
    */
    bool TrackReferenceKeyFrame();

    /**
     * 更新 mLastFrame
     * 更新 mlpTemporalPoints
     */
    void UpdateLastFrame();

    /**
      * @brief 根据匀速度模型对上一帧mLastFrame的MapPoints与当前帧mCurrentFrame进行特征点 **跟踪匹配**
      * 1. 非单目情况，需要对上一帧产生一些新的MapPoints（临时）
      * 2. 将上一帧的MapPoints投影到当前帧的图像平面上，在投影的位置进行区域 **匹配**
      * 3. 根据匹配优化当前帧的 **姿态**
      * 4. 根据姿态剔除误匹配
      * @return 如果匹配数大于10，返回true
      */    
    bool TrackWithMotionModel();

    // BOW搜索候选关键帧，PnP求解 **位姿**
    bool Relocalization();

    // 更新局部地图，即 更新 **局部地图keyframe** + **局部地图mappoint**
    void UpdateLocalMap();          // keyframe + mappoint

    // 将 mvpLocalKeyFrames 中的mappoint，添加到局部地图关键点 mvpLocalMapPoints 中
    void UpdateLocalPoints();       // Local MapPoints

    /**
     * 更新 mpReferenceKF，mCurrentFrame.mpReferenceKF
     * 更新 局部地图关键帧 mvpLocalKeyFrames
     */
    void UpdateLocalKeyFrames();    // ReferenceKF + LocalKeyFrames

    /**
    * @brief 对 mvpLocalKeyFrames ， mvpLocalMapPoints 进行跟踪
    * 
    * 1. 更新 局部地图，包括局部关键帧和关键点
    * 2. 以局部地图的mappoint为范围和当前帧进行特征匹配
    * 3. 根据匹配对通过BA估计 **当前帧的姿态**
    * 4. 更新 当前帧的MapPoints被观测程度 ，统计 跟踪局部地图的效果
    * @return 根据跟踪局部地图的效果判断当前帧的跟踪成功与否，返回其判断结果
    * @see V-D track Local Map
    */
    bool TrackLocalMap();

    // 在局部地图的mappoint中查找在当前帧视野范围内的点，将视野范围内的点和当前帧的特征点进行投影 **匹配**
    void SearchLocalPoints();

    // 判断是否需要添加新的keyframe
    bool NeedNewKeyFrame();

    /**
    * @brief 创建新的关键帧
    *
    * 对于非单目的情况，同时创建新的MapPoints
    */
    void CreateNewKeyFrame();

    // In case of performing only localization, this flag is true when there are no matches to
    // points in the map. Still tracking will continue if there are enough matches with temporal points.
    // In that case we are doing visual odometry. The system will try to do relocalization to recover
    // "zero-drift" localization to the map.
    // mbVO是mbOnlyTracking为true时的才有的一个变量
    // mbVO为false表示此帧匹配了很多的MapPoints，跟踪很正常，
    // mbVO为true表明此帧匹配了很少的MapPoints，少于10个，要跪的节奏
    bool mbVO;

    // Other Thread Pointers
    LocalMapping* mpLocalMapper;
    LoopClosing* mpLoopClosing;

    // ORB
    ORBextractor* mpORBextractorLeft, *mpORBextractorRight;
    ORBextractor* mpIniORBextractor;

    // BoW
    ORBVocabulary* mpORBVocabulary;
    KeyFrameDatabase* mpKeyFrameDB;

    // Initalization (only for monocular)
    Initializer* mpInitializer;

    // Local Map
    // 参考关键帧
    // 在 CreateNewKeyFrame() 中，为当前帧
    // 在 UpdateLocalKeyFrames() 中，为当前帧共视程度最高的关键帧, it maybe the most robust keyframe
    KeyFrame* mpReferenceKF;
    std::vector<KeyFrame*> mvpLocalKeyFrames;
    // mvpLocalKeyFrames 的 所有关键帧的所有匹配的 mappoint集合
    std::vector<MapPoint*> mvpLocalMapPoints;
    
    // System
    System* mpSystem;
    
    // Drawers
    Viewer* mpViewer;
    FrameDrawer* mpFrameDrawer;
    MapDrawer* mpMapDrawer;

    // Map
    Map* mpMap;

    // Calibration matrix
    cv::Mat mK;
    cv::Mat mDistCoef;
    float mbf;

    // New KeyFrame rules (according to fps)
    int mMinFrames;
    int mMaxFrames;

    // Threshold close/far points
    // Points seen as close by the stereo/RGBD sensor are considered reliable
    // and inserted from just one frame. Far points require a match in two keyframes.
    float mThDepth;

    // For RGB-D inputs only. For some datasets (e.g. TUM) the depthmap values are scaled.
    float mDepthMapFactor;

    // Current matches in frame
    // 当前的匹配
    int mnMatchesInliers;

    // Last Frame, KeyFrame and Relocalisation Info
    KeyFrame* mpLastKeyFrame;           // 最近新插入的keyframe
    Frame mLastFrame;                   // 记录最近的一帧
    unsigned int mnLastKeyFrameId;      // tracking 上一次插入 mpLastKeyFrame 的 Frame ID
    unsigned int mnLastRelocFrameId;    // 上一次 Relocalization() 使用的 Frame ID ，最近一次重定位帧的 ID

    // Motion Model
    cv::Mat mVelocity;

    // Color order (true RGB, false BGR, ignored if grayscale)
    bool mbRGB;

    // 在 UpdateLastFrame() 更新
    list<MapPoint*> mlpTemporalPoints;
};

} //namespace ORB_SLAM

#endif // TRACKING_H
