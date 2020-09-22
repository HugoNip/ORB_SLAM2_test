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


#include "Tracking.h"

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include"ORBmatcher.h"
#include"FrameDrawer.h"
#include"Converter.h"
#include"Map.h"
#include"Initializer.h"

#include"Optimizer.h"
#include"PnPsolver.h"

#include<iostream>

#include<mutex>


using namespace std;

namespace ORB_SLAM2
{

Tracking::Tracking(System *pSys, ORBVocabulary* pVoc, FrameDrawer *pFrameDrawer, 
                   MapDrawer *pMapDrawer, Map *pMap, KeyFrameDatabase* pKFDB, 
                   const string &strSettingPath, const int sensor):
                   mState(NO_IMAGES_YET), mSensor(sensor), mbOnlyTracking(false), mbVO(false), 
                   mpORBVocabulary(pVoc), mpKeyFrameDB(pKFDB), mpInitializer(static_cast<Initializer*>(NULL)), 
                   mpSystem(pSys), mpViewer(NULL), mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), 
                   mpMap(pMap), mnLastRelocFrameId(0) {
    // Load camera parameters from settings file

    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    // for undistort
    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];   // baseline

    float fps = fSettings["Camera.fps"];
    if(fps==0)
        fps=30;

    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;
    mMaxFrames = fps;

    cout << endl << "Camera Parameters: " << endl;
    cout << "- fx: " << fx << endl;
    cout << "- fy: " << fy << endl;
    cout << "- cx: " << cx << endl;
    cout << "- cy: " << cy << endl;
    cout << "- k1: " << DistCoef.at<float>(0) << endl;
    cout << "- k2: " << DistCoef.at<float>(1) << endl;
    if(DistCoef.rows==5)
        cout << "- k3: " << DistCoef.at<float>(4) << endl;
    cout << "- p1: " << DistCoef.at<float>(2) << endl;
    cout << "- p2: " << DistCoef.at<float>(3) << endl;
    cout << "- fps: " << fps << endl;


    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if(mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    // Load ORB parameters
    int nFeatures       = fSettings["ORBextractor.nFeatures"];
    float fScaleFactor  = fSettings["ORBextractor.scaleFactor"];
    int nLevels         = fSettings["ORBextractor.nLevels"];
    int fIniThFAST      = fSettings["ORBextractor.iniThFAST"];
    int fMinThFAST      = fSettings["ORBextractor.minThFAST"];

    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(sensor==System::STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(sensor==System::MONOCULAR)
        mpIniORBextractor = new ORBextractor(2*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    cout << endl  << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

    if(sensor==System::STEREO || sensor==System::RGBD)
    {
        mThDepth = mbf*(float)fSettings["ThDepth"]/fx;
        cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
    }

    if(sensor==System::RGBD)
    {
        mDepthMapFactor = fSettings["DepthMapFactor"];
        if(fabs(mDepthMapFactor)<1e-5)
            mDepthMapFactor=1;
        else
            mDepthMapFactor = 1.0f/mDepthMapFactor;
    }

}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing=pLoopClosing;
}

void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer=pViewer;
}


// image preprocessing
cv::Mat Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp) 
{
    mImGray = imRectLeft;
    cv::Mat imGrayRight = imRectRight;

    // convert to GRAY
    if(mImGray.channels()==3)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGB2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGR2GRAY); // channel BGR
            cvtColor(imGrayRight,imGrayRight,CV_BGR2GRAY);
        }
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGBA2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGRA2GRAY);
        }
    }

    // !!! compute ORB
    mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}


cv::Mat Tracking::GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp)
{
    mImGray = imRGB;
    cv::Mat imDepth = imD;

    // convert to GRAY
    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    if((fabs(mDepthMapFactor-1.0f)>1e-5) || imDepth.type()!=CV_32F)
        imDepth.convertTo(imDepth,CV_32F,mDepthMapFactor);

    // !!! compute ORB
    mCurrentFrame = Frame(mImGray,imDepth,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}


cv::Mat Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp)
{
    mImGray = im;

    // convert to GRAY
    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    // !!! compute ORB
    if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET)
        mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);
    else
        mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}


/** 
 * @brief
 * Main tracking function. It is independent of the input sensor.
 * 
 * 初始化之后ORB-SLAM有三种跟踪模型可供选择
 * a.TrackWithMotionModel();    运动模型：  根据运动模型估计当前帧位姿——根据匀速运动模型对**上一帧**的地图点进行跟踪——优化位姿
 * b.TrackReferenceKeyFrame();  参考帧模型： BoW搜索当前帧与**参考帧**的匹配点——将**上一帧**的位姿作为当前帧的**初始值**——通过优化3D-2D的重投影误差来获得位姿
 * c.Relocalization()；         重定位模型： 计算当前帧的BoW——检测满足重定位条件的**候选帧**——通过BoW搜索当前帧与候选帧的匹配点——大于15个点就进行PnP位姿估计——优化
 * 
 * 这三个模型的选择方法：
 * 首先假设相机恒速（即Rt和上一帧相同），选择**运动模型**, 然后计算匹配点数（如果匹配足够多则认为跟踪成功），
 * 如果匹配点数目较少，说明**运动模型**失效， 则选择**参考帧模型**（即特征匹配，PnP求解），
 * 如果参考帧模型同样不能进行跟踪，说明两帧键没有相关性，这时需要选择**重定位模型**，
 * 即和已经产生的**关键帧**中进行匹配（看看是否到了之前已经到过的地方）确定相机位姿，
 * 如果重定位仍然不能成功，则说明跟踪彻底丢失，要么等待相机回转，要不进行重置。
 */
void Tracking::Track() 
{
    // 如果图像复位过、或者第一次运行，则为 NO_IMAGES_YET 状态
    if(mState==NO_IMAGES_YET)
    {
        mState = NOT_INITIALIZED;
    }

    mLastProcessedState=mState;

    // Get Map Mutex -> Map cannot be changed
    unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

    // 如果tracking没有初始化 ，则**初始化**
    if(mState==NOT_INITIALIZED)
    {
        if(mSensor==System::STEREO || mSensor==System::RGBD)
            StereoInitialization();
        else
            MonocularInitialization();

        mpFrameDrawer->Update(this);

        if(mState!=OK)
            return;
    }
    else
    {
        // System is **initialized**. Track Frame.
        // bOK为临时变量，bOK==true现在tracking状态正常,能够及时的反应现在tracking的状态
        bool bOK;

        // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
        // 用户可以通过在viewer中的开关 menuLocalizationMode ，控制是否 ActivateLocalizationMode ，并最终管控 mbOnlyTracking 是否为 true
        // mbOnlyTracking 等于false表示 正常VO模式 （有地图更新）， mbOnlyTracking 等于true表示 用户手动选择 定位模式
        //
        // 局部地图激活(!mbOnlyTracking)：
        // 如果(mState==OK)，CheckReplacedInLastFrame()首先更新上一帧被替换的MapPoints，
        // 然后如果特征点匹配太少，需要匹配参考关键帧 bOK = TrackReferenceKeyFrame() 
        // 否则根据匀速运动模型匹配bOK = TrackWithMotionModel()。
        // 特殊情况初始化跟踪失败需要重定位bOK = Relocalization();
        if(!mbOnlyTracking)
        {
            // Local Mapping is activated. This is the normal behaviour, unless
            // you explicitly activate the "only tracking" mode.
            // **正常VO模式（有地图更新）**

            if(mState==OK)  // **tracking is good**
            {
                // Local Mapping might have changed some MapPoints tracked in last frame
                // lastframe 中可以看到的 mappoint 替换为 lastframe 储存的备胎 mappoint 点 mpReplaced ，也就是更新 mappoint
                CheckReplacedInLastFrame();

                // 运动模型是空的或刚完成重定位
                if(mVelocity.empty() || mCurrentFrame.mnId<mnLastRelocFrameId+2)
                {
                    /**
                     * 将 参考帧关键帧的位姿 作为当前帧的 初始位姿 进行跟踪；
                     * 匹配参考帧关键帧中有对应mappoint的特征点与当前帧特征点，通过dbow加速匹配；
                     * 优化3D点重投影误差，得到更精确的位姿以及剔除错误的特征点匹配；
                     * @return 如果匹配数大于10，返回true
                     */
                    bOK = TrackReferenceKeyFrame();
                }
                else        // **tracking is lost**
                {
                    bOK = TrackWithMotionModel();
                    if(!bOK)
                        bOK = TrackReferenceKeyFrame();
                }
            }
            else
            {
                // BOW搜索候选关键帧，PnP求解位姿
                bOK = Relocalization();
            }
        }
        else    // 只进行跟踪tracking，局部地图不工作
        {
            // Localization Mode: Local Mapping is deactivated
            // **用户手动选择 定位模式**

            if(mState==LOST) // **tracking is lost**
            {
                bOK = Relocalization();
            }
            else             // **tracking is good**
            {
                // mbVO是mbOnlyTracking为true时的才有的一个变量
                // mbVO为false表示此帧匹配了很多的MapPoints，跟踪很正常，
                // mbVO为true表明此帧匹配了很少的MapPoints，少于10个，要跪的节奏
                if(!mbVO)   // mbVO==false
                {
                    // In last frame we tracked enough MapPoints in the map
                    if(!mVelocity.empty())  // 上一帧有速度，跟踪模型
                    {
                        bOK = TrackWithMotionModel();
                    }
                    else                    // 上一帧没速度，跟踪关键帧
                    {
                        bOK = TrackReferenceKeyFrame();
                    }
                }
                else
                {
                    // In last frame we tracked mainly "visual odometry" points.

                    // We compute two camera poses, one from motion model (TrackWithMotionModel()) 
                    // and one doing relocalization (TrackReferenceKeyFrame()).
                    // If relocalization is sucessfull we choose that solution, otherwise we retain
                    // the "visual odometry" solution.
                    // 我们跟踪和重定位都计算，如果重定位成功就选择重定位来计算位姿
                    // 先使用运动模型和重定位计算两种相机位姿，如果重定位失败，保持VO结果
                    // mbVO 为1，则表明此帧匹配了很少的3D map点，少于10个，要跪的节奏，既做跟踪又做定位
                    bool bOKMM = false;         // 运动模型 是否成功判断标志
                    bool bOKReloc = false;      // 重定位 是否成功判断标志
                    vector<MapPoint*> vpMPsMM;  // 记录地图点
                    vector<bool> vbOutMM;       // 记录外点
                    cv::Mat TcwMM;              // 变换矩阵
                    if(!mVelocity.empty())      // 如果运动模型非空, 有速度
                    {
                        // 使用运动模型进行跟踪
                        bOKMM = TrackWithMotionModel();
                        vpMPsMM = mCurrentFrame.mvpMapPoints;
                        vbOutMM = mCurrentFrame.mvbOutlier;
                        TcwMM = mCurrentFrame.mTcw.clone(); // 当前帧的变换矩阵
                    }
                    // 使用重定位计算位姿
                    bOKReloc = Relocalization();    // Relocalization is successful

                    
                    if(bOKMM && !bOKReloc)  // 重定位没有成功，但是运动模型跟踪成功
                    {
                        mCurrentFrame.SetPose(TcwMM);
                        mCurrentFrame.mvpMapPoints = vpMPsMM;
                        mCurrentFrame.mvbOutlier = vbOutMM;

                        // 确认mbVO==1
                        if(mbVO)
                        {
                            // 更新当前帧的MapPoints被观测程度
                            for(int i =0; i<mCurrentFrame.N; i++)
                            {
                                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                                {
                                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound(); // can be found and has corresponding keypoint
                                }
                            }
                        }
                    }
                    else if(bOKReloc)       // 只要重定位成功整个跟踪过程正常进行（定位与跟踪，更相信重定位）
                    {
                        mbVO = false;
                    }

                    bOK = bOKReloc || bOKMM;
                }
            }
        }

        // 将最新的关键帧作为reference frame
        mCurrentFrame.mpReferenceKF = mpReferenceKF;

        // If we have an initial estimation of the camera pose and matching. Track the local map.
        if(!mbOnlyTracking)
        {
            if(bOK)
                bOK = TrackLocalMap();
        }
        else
        {
            // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
            // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
            // the camera we will use the local map again.
            if(bOK && !mbVO)
                bOK = TrackLocalMap();
        }

        if(bOK)
            mState = OK;
        else
            mState=LOST;

        // Update drawer
        mpFrameDrawer->Update(this);

        // If tracking were good, check if we insert a keyframe
        if(bOK)
        {
            // Update motion model
            if(!mLastFrame.mTcw.empty())
            {
                // 更新恒速运动模型 TrackWithMotionModel 中的 mVelocity
                cv::Mat LastTwc = cv::Mat::eye(4,4,CV_32F); // Twc, SE3
                mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0,3).colRange(0,3));
                mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0,3).col(3));
                mVelocity = mCurrentFrame.mTcw*LastTwc; // vector
            }
            else
                mVelocity = cv::Mat();

            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

            // Clean VO matches
            // 步骤2.4：清除 UpdateLastFrame() 中为当前帧临时添加的 MapPoints
            for(int i=0; i<mCurrentFrame.N; i++)
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(pMP)
                    if(pMP->Observations()<1)
                    {
                        mCurrentFrame.mvbOutlier[i] = false;
                        mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                    }
            }

            // Delete temporal MapPoints
            // 清除临时的MapPoints，这些MapPoints在 TrackWithMotionModel 的 UpdateLastFrame 函数里生成（仅双目和rgbd）
            // 步骤2.4中只是在当前帧中将这些MapPoints剔除，这里从MapPoints数据库中删除
            // 这里生成的仅仅是为了提高双目或rgbd摄像头的帧间跟踪效果，用完以后就扔了，没有添加到地图中
            for(list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(), lend =  mlpTemporalPoints.end(); lit!=lend; lit++)
            {
                MapPoint* pMP = *lit;
                delete pMP;
            }
            // 这里不仅仅是清除 mlpTemporalPoints ，通过delete pMP还删除了指针指向的MapPoint
            mlpTemporalPoints.clear();

            // Check if we need to insert a new keyframe
            // 判断是否插入keyframe
            if(NeedNewKeyFrame())
                CreateNewKeyFrame();

            // We allow points (i.e., MapPoints) with high innovation (considererd outliers by the Huber Function)
            // pass to the new keyframe, so that bundle adjustment will finally decide
            // if they are **outliers** or not. We don't want next frame to estimate its position
            // with those points so we discard them in the frame.
            // 删除那些在bundle adjustment中检测为outlier的3D MapPoiint
            for(int i=0; i<mCurrentFrame.N;i++)
            {
                if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                    mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
            }
        }

        // Reset if the camera get lost soon after initialization
        // 如果mState==LOST，且mpMap->KeyFramesInMap()<=5（说明刚刚初始化），则reset；
        // 如果mState==LOST，且mpMap->KeyFramesInMap()>5，则会在下一帧中执行bOK = Relocalization();
        if(mState==LOST)
        {
            if(mpMap->KeyFramesInMap()<=5)
            {
                cout << "Track lost soon after initialisation, reseting..." << endl;
                mpSystem->Reset();
                return;
            }
        }

        if(!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

        // 保存上一帧的数据
        mLastFrame = Frame(mCurrentFrame);
    }

    // 记录位姿信息，用于轨迹复现
    // Store frame pose information to retrieve the complete camera trajectory afterwards.
    if(!mCurrentFrame.mTcw.empty())
    {
        cv::Mat Tcr = mCurrentFrame.mTcw*mCurrentFrame.mpReferenceKF->GetPoseInverse();
        mlRelativeFramePoses.push_back(Tcr);
        mlpReferences.push_back(mpReferenceKF);
        mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
        mlbLost.push_back(mState==LOST);
    }
    else
    {
        // This can happen if tracking is lost
        mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
        mlpReferences.push_back(mlpReferences.back());
        mlFrameTimes.push_back(mlFrameTimes.back());
        mlbLost.push_back(mState==LOST);
    }

}


void Tracking::StereoInitialization()
{
    if(mCurrentFrame.N>500)
    {
        // Set Frame pose to the origin
        /**
         * T
         * [1 0 0 0]
         * [0 1 0 0]
         * [0 0 1 0]
         * [0 0 0 1]
         */
        mCurrentFrame.SetPose(cv::Mat::eye(4,4,CV_32F));

        // Create KeyFrame
        // 当前帧被设置为第一帧
        KeyFrame* pKFini = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

        // Insert KeyFrame in the map
        // 当前帧构造为关键帧
        mpMap->AddKeyFrame(pKFini);     // Map* mpMap; add pKF into mspKeyFrames

        // Create MapPoints and asscoiate to KeyFrame
        // 一个初始的地图（很稀疏的点云）将建立起来
        for(int i=0; i<mCurrentFrame.N;i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);     // world coordinate
                // MapPoint::MapPoint(const cv::Mat &Pos, KeyFrame *pRefKF, Map* pMap)
                MapPoint* pNewMP = new MapPoint(x3D,pKFini,mpMap);
                pNewMP->AddObservation(pKFini,i);
                pKFini->AddMapPoint(pNewMP,i);
                pNewMP->ComputeDistinctiveDescriptors();
                pNewMP->UpdateNormalAndDepth();
                mpMap->AddMapPoint(pNewMP);

                mCurrentFrame.mvpMapPoints[i]=pNewMP;
            }
        }

        cout << "New map created with " << mpMap->MapPointsInMap() << " points" << endl;

        /**
         * 相关后处理
         * （1）首先第一个关键帧得送到局部建图线程中去。
         */
        mpLocalMapper->InsertKeyFrame(pKFini);

        /**
         * （2）更新Tracking类的“上一…”信息：
         *      当前帧通过一个Frame构造函数初始并设置为Tracking类的上一帧mLastFrame，
         *      当前帧的ID（mnID）设置为上一个关键帧的ID（mnLastKeyFrameId），
         *      构造的第一个关键帧记录为上一个关键帧（mpLastKeyFrame）。
         */
        mLastFrame = Frame(mCurrentFrame);
        mnLastKeyFrameId=mCurrentFrame.mnId;            // idx
        mpLastKeyFrame = pKFini;

        /**
         * （3）Tracking类的局部地图（不是地图）的相关数据记录：
         *      初始关键帧推入mvpLocalKeyFrames 保存，
         *      当前已有的初始点云保存为 mvpLocalMapPoints，
         *      初始关键帧保存为当前Tracking的参考关键帧成员 mpReferenceKF
         */
        mvpLocalKeyFrames.push_back(pKFini);            // Local Map
        mvpLocalMapPoints=mpMap->GetAllMapPoints();     // Local Map
        mpReferenceKF = pKFini;                         // Local Map
        mCurrentFrame.mpReferenceKF = pKFini;           // Current Frame

        /**
         * （4）当前帧的参考关键帧设为根据自己创建的初始关键帧，
         *      地图的参考地图点 mvpReferenceMapPoints 设置为前面保存的局部地图点，
         *      初始关键帧被推入地图的 mvpKeyFrameOrigins 保存
         */
        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);    // -> mvpReferenceMapPoints
        mpMap->mvpKeyFrameOrigins.push_back(pKFini);

        /**
         * （5）当前帧的位姿Tcw作为当前相机位姿传递给MapDrawer
         */
        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

        /**
         * （6）Tracking的状态eState设为OK
         */
        mState=OK;
    }
}


/**
 * @brief 
 * 单目相机的初始化过程，通过将 最初的 两帧 之间进行 对极约束 和 全局BA 优化，得到较为准确的初始值
 * 
 * 并行地计算基础矩阵和单应性矩阵，选取其中一个模型，恢复出最开始两帧之间的相对姿态以及点云
 * 得到初始两帧的匹配、相对运动、初始MapPoints
 */
void Tracking::MonocularInitialization()
{
    // 如果单目初始器还没有被创建，则创建单目初始器
    if(!mpInitializer)  // no Initializer
    {
        // Set Reference Frame
        // 如果当前帧特征点数量大于100
        if(mCurrentFrame.mvKeys.size()>100)
        {
            // 1）当第一次进入该方法的时候，没有先前的帧数据，将当前帧保存为**初始帧**和**最后一帧**，并初始化一个**初始化器**
            mInitialFrame = Frame(mCurrentFrame);   // first rame
            mLastFrame = Frame(mCurrentFrame);      // last frame

            /**
             * mInitialFrame 中待匹配的特征点的 像素位置
             * std::vector<cv::Point2f> mvbPrevMatched;    // [u, v] keypoints
             * mvbPrevMatched最大的情况就是所有特征点都被跟踪上
             */
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());       // Prev: 待匹配
            for(size_t i=0; i<mCurrentFrame.mvKeysUn.size(); i++)
                mvbPrevMatched[i]=mCurrentFrame.mvKeysUn[i].pt;         // 待匹配的特征点
            
            // 确认mpInitializer指向NULL
            if(mpInitializer)
                delete mpInitializer;

            mpInitializer =  new Initializer(mCurrentFrame,1.0,200);    // 初始化一个**初始化器**

            /**
             * 将mvIniMatches全部初始化为-1 (-1: unmatch)
             * 初始化时 得到的 特征点匹配，大小是 mInitialFrame/ReferenceFrame 1 的特征点数量
             * 其值是 CurrentFrame 2 特征点**序号(idx)**
             */
            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);           

            return;
        }
    }
    else
    {
        /**
         * 如果指针mpInitializer没有指向NULL，也就是说我们在之前已经通过mInitialFrame
         * 新建过一个Initializer对象，并使 mpInitializer 指向它了
         * 2）第二次进入该方法的时候，已经有初始化器了
         */

        /**
         * Try to initialize
         * 尝试初始化
         * 如果当前帧特征点数量<=100
         * 特征点数过少，则删除当前的初始化器
         * unqualified
         */
        if((int)mCurrentFrame.mvKeys.size()<=100)
        {
            // 删除mpInitializer指针并指向NULL
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);        // reset an initializer
            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);
            return;
        }

        /**
         * 3）利用 ORB匹配器 ，对 当前帧 和 初始帧 进行匹配，对应关系 小于 100个 时失败
         * Find correspondences
         * 新建一个 ORBmatcher 对象
         * 寻找特征点匹配
         */
        ORBmatcher matcher(0.9,true);
        int nmatches = matcher.SearchForInitialization(mInitialFrame,mCurrentFrame,mvbPrevMatched,mvIniMatches,100);

        // Check if there are enough correspondences
        // 如果匹配的点过少，则删除当前的初始化器
        if(nmatches<100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            return;
        }

        /**
         * 4）利用 八点法 的 对极约束，启动两个线程分别计算 单应矩阵 和 基础矩阵 ，
         * 并通过 score 判断用单应矩阵恢复运动轨迹还是使用基础矩阵恢复运动轨迹
         * ========================================================
         * 到这里说明初始化条件已经成熟
         * 特征点数&&匹配的点
         * 开始要计算位姿R，t了！！
         */
        cv::Mat Rcw;                    // Current Camera Rotation
        cv::Mat tcw;                    // Current Camera Translation
        // 初始化成功后，匹配点中三角化投影成功的情况
        vector<bool> vbTriangulated;    // Triangulated Correspondences (mvIniMatches)

        // 通过H模型或F模型进行单目初始化，得到两帧间**相对运动R+t**、初始**MapPoints**
        if(mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated)) // Rcw, tcw
        {
            // ReconstructH，或者ReconstructF中解出Rt后，会有一些点不能三角化重投影成功。
            // 在根据 vbTriangulated 中特征点三角化投影成功的情况，**去除一些匹配点**
            for(size_t i=0, iend=mvIniMatches.size(); i<iend;i++)
            {
                if(mvIniMatches[i]>=0 && !vbTriangulated[i])
                {
                    mvIniMatches[i]=-1;
                    nmatches--;
                }
            }

            // Set Frame Poses
            // 初始化 mInitialFrame 的**位姿**
            // 将 mInitialFrame 位姿设置为**世界坐标**
            mInitialFrame.SetPose(cv::Mat::eye(4,4,CV_32F));
            cv::Mat Tcw = cv::Mat::eye(4,4,CV_32F);
            Rcw.copyTo(Tcw.rowRange(0,3).colRange(0,3));
            tcw.copyTo(Tcw.rowRange(0,3).col(3));
            mCurrentFrame.SetPose(Tcw);

            /**
             * 5）将 初始帧(mInitialFrame) 和 当前帧(mCurrentFrame) 创建为**关键帧**，并创建地图点**MapPoint**
             * 6）更新**共视关系**, 通过 全局 BundleAdjustment 优化相机位姿 和 关键点 坐标
             * 7）设置 单位深度 并 缩放 initial baseline 和 地图点
             * 8）其他变量的初始化
             */
            CreateInitialMapMonocular();
        }
    }
}


/**
 * @brief 单目模式下初始化后，开始建图
 */
void Tracking::CreateInitialMapMonocular()
{
    // Create KeyFrames
    // 5）将 初始帧(mInitialFrame) 和 当前帧(mCurrentFrame) 创建为**关键帧**，并创建地图点**MapPoints**
    KeyFrame* pKFini = new KeyFrame(mInitialFrame,mpMap,mpKeyFrameDB);
    KeyFrame* pKFcur = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

    // 计算关键帧的词袋bow和featurevector
    pKFini->ComputeBoW();
    pKFcur->ComputeBoW();

    // Insert KFs in the map
    mpMap->AddKeyFrame(pKFini);
    mpMap->AddKeyFrame(pKFcur);

    // Create MapPoints and asscoiate to keyframes
    // 遍历每一个初始化时得到的特征点匹配,将三角化**重投影成功**的特征点转化为**MapPoint**
    /**
     * 将 mvIniMatches 全部初始化为-1 (-1: unmatch)
     * 初始化时 得到的 特征点匹配
     * 大小是 mInitialFrame/ReferenceFrame 1 的特征点数量
     * 其值是 CurrentFrame 2 特征点**序号(idx)**
     */
    for(size_t i=0; i<mvIniMatches.size();i++)                  // i:               idx in mInitialFrame
    {                                                           // mvIniMatches[i]: idx in CurrentFrame
        if(mvIniMatches[i]<0)   // -1: unmatch
            continue;

        // Create MapPoint.
        // mvIniP3D: 初始化时 三角化投影成功的匹配点对应的 3d点
        cv::Mat worldPos(mvIniP3D[i]);                          // world coordinate
        // 新建mappoint对象，注意mappoint的参考帧是**pKFcur**
        MapPoint* pMP = new MapPoint(worldPos,pKFcur,mpMap);

        // 给关键帧(pKFini,pKFcur)添加mappoint，让keyframe知道自己可以看到哪些mappoint
        pKFini->AddMapPoint(pMP,i);
        pKFcur->AddMapPoint(pMP,mvIniMatches[i]);

        // 让pMP知道自己可以被pKFini，pKFcur看到
        pMP->AddObservation(pKFini,i);
        pMP->AddObservation(pKFcur,mvIniMatches[i]);

        // each mappoint has its own descriptor (the best one)
        // 找出最能代表此mappoint的**描述子**
        pMP->ComputeDistinctiveDescriptors();                   // Descriptor
        // 更新此mappoint参考帧光心到mappoint**平均观测方向**以及**观测距离范围**
        pMP->UpdateNormalAndDepth();

        // Fill Current Frame structure
        // 更新当前帧Frame能看到哪些mappoint
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        // 标记当前帧的特征点不是**Outlier**
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

        // Add this MapPoint into Map (Global)
        mpMap->AddMapPoint(pMP);
    }

    // Update Connections
    /**
     * 6）更新**共视关系**, 通过 全局 BundleAdjustment 优化相机位姿 和 关键点 坐标
     * 
     * **共视关系**: Covisibility graph
     *              essential graph
     *              spanningtree
     */
    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    // Bundle Adjustment
    cout << "New Map created with " << mpMap->MapPointsInMap() << " points" << endl;

    Optimizer::GlobalBundleAdjustemnt(mpMap, 20);

    // Set median depth to 1
    /**
     * 7）设置 单位深度 并 缩放 initial baseline 和 MapPoints
     * 
     * 返回mappoint集合在此帧的深度的中位数
     * 将MapPoints的**中值深度**归一化到1，并归一化**两帧之间变换**
     * 评估关键帧场景深度，q=2表示中值
     */
    float medianDepth = pKFini->ComputeSceneMedianDepth(2);
    float invMedianDepth = 1.0f/medianDepth;

    if(medianDepth<0 || pKFcur->TrackedMapPoints(1)<100)
    {
        cout << "Wrong initialization, reseting..." << endl;
        Reset();
        return;
    }

    // Scale initial baseline = baseline * invMedianDepth
    cv::Mat Tc2w = pKFcur->GetPose(); // Tcw
    // 利用invMedianDepth将z归一化到1
    Tc2w.col(3).rowRange(0,3) = Tc2w.col(3).rowRange(0,3)*invMedianDepth;
    pKFcur->SetPose(Tc2w);

    // Scale MapPoints = (X, Y, Z) * invMedianDepth
    // 把3D点的尺度也归一化到1
    vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
    for(size_t iMP=0; iMP<vpAllMapPoints.size(); iMP++)
    {
        if(vpAllMapPoints[iMP])
        {
            MapPoint* pMP = vpAllMapPoints[iMP];
            pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);
        }
    }

    /**
     * 8）其他变量的初始化
     * 
     * 向 mpLocalMapper 插入关键帧 pKFini，pKFcur
     */
    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);

    mCurrentFrame.SetPose(pKFcur->GetPose());
    mnLastKeyFrameId=mCurrentFrame.mnId;
    mpLastKeyFrame = pKFcur;

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints=mpMap->GetAllMapPoints();
    mpReferenceKF = pKFcur;
    mCurrentFrame.mpReferenceKF = pKFcur;

    mLastFrame = Frame(mCurrentFrame);

    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

    mpMap->mvpKeyFrameOrigins.push_back(pKFini);

    mState=OK;
}


/**
 * ocal Mapping线程可能会将关键帧中某些MapPoints进行替换，
 * 由于tracking中需要用到mLastFrame，这里检查并更新上一帧中被替换的MapPoints
 */
void Tracking::CheckReplacedInLastFrame()
{
    for(int i =0; i<mLastFrame.N; i++)
    {
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];

        if(pMP)
        {
            MapPoint* pRep = pMP->GetReplaced();
            if(pRep)
            {
                mLastFrame.mvpMapPoints[i] = pRep;
            }
        }
    }
}


/**
 * @brief 按照关键帧来进行 Track ，从关键帧中查找 Bow 相近的帧，进行 匹配 优化位姿   
 * 
 * 按照关键帧进行 Track 的方法和 运动模式 恢复相机运动位姿的方法接近   
 * 1）首先求解当前帧的 BOW向量
 * 2）再搜索 当前帧 和 关键帧 之间的 关键点匹配关系 ，如果这个匹配关系小于15对的话， Track失败
 * 3）接着,讲当前帧的位置假定到上一帧的位置那里
 * 4）并通过最小二乘法 优化 相机的位姿
 * 5）最后依然是抛弃无用的杂点，当 match数 大于等于10的时候，返回true成功
 * 
 * 
 * 步骤1：将当前帧(cur frames)的描述子转化为BoW向量。 
 * 步骤2：利用 BOW方法 对ORB特征点进行 匹配, 匹配参考帧关键帧中有对应mappoint的特征点与当前帧特征点，通过dbow加速匹配
 * 
 * (for step 1 + 2) 
 * 在已经获得图像特征点集合的基础上, 再根据词典, 对每个特征做一次分类.
 * 再对第二幅图像提取特征, 然后也根据词典, 也对这幅图像的所有特征进行分类
 * 用分类后的特征类别代替原本的特征 descriptor , 即用 一个数字代替一个向量 进行比对, 显然速度可以大大提升
 * 也就是说之前用若干个ORB特征点代表一副图像，根据词典, 对每个特征做一次分类之后，
 * 是用若干个单词代表一副图像（一个单词包含数个相似的特征点）
 * 将两幅图像分别用a和b个单词代表之后，在单词的基础上进行匹配
 * 之后在任意一对匹配好的单词中做 ORB特征点匹配
 * 对所有相互匹配的 单词对 中进行ORB特征点匹配之后就完成了整副图像的ORB特征点匹配
 * 这样做相对于暴力匹配减少了计算量
 * 
 * 步骤3: 将**上一帧的位姿**作为**当前帧位姿的初始值**，之后将其带入优化模型中。
 * 
 * keyframe(KF)的内容：
 * mappoints（地图点）
 * keypoints（特征点）
 * TCW（位姿）
 * 
 * cur frame 的内容： 
 * mappoints（地图点）  取自KF 和KF 的mappoints相同
 * keypoints（特征点）  通过与KF中ORB特征点匹配获得（利用BOW方法加速）
 * TCW（位姿）          初始值设置与上一个 frame 相同
 * 
 * 步骤4: 这时设KF为世界坐标，cur frame 可以通过优化3D-2D的重投影误差(PnP)来获得 cur frame 的位姿TCW,
 *       也就是Optimizer::PoseOptimization(&mCurrentFrame)函数
 * 
 * 步骤5：剔除优化后的outlier匹配点（MapPoints）
 * 
 * TrackWithMotionModel() 和 TrackReferenceKeyFrame()都是通过输入的帧（cur frame）
 * 计算cur frame 的R，T。不同之处 是二者ORB特征点匹配的方法不同。
 * 
 * @return 如果匹配数大于10，返回true
 */
bool Tracking::TrackReferenceKeyFrame()
{
    // Compute Bag of Words vector
    // 计算当前帧的Bow向量
    mCurrentFrame.ComputeBoW();

    // We perform first an ORB matching with the reference keyframe
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.7,true);
    vector<MapPoint*> vpMapPointMatches;

    // 通过特征点的BoW**加速匹配**当前帧与参考关键帧之间的**特征点**
    // int ORBmatcher::SearchByBoW(KeyFrame* pKF,Frame &F, vector<MapPoint*> &vpMapPointMatches)
    int nmatches = matcher.SearchByBoW(mpReferenceKF,mCurrentFrame,vpMapPointMatches);

    // 匹配数小于15，表示跟踪失败
    if(nmatches<15)
        return false;

    mCurrentFrame.mvpMapPoints = vpMapPointMatches;
    // 将上一帧的位姿作为当前帧位姿的初始值，在PoseOptimization可以收敛快一些
    mCurrentFrame.SetPose(mLastFrame.mTcw);

    // 通过优化3D-2D的重投影误差来获得位姿
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    // 剔除优化后的outlier匹配点
    // 遍历mCurrentFrame每个特征点
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        // 如果这个特征点与相对应的mappoint
        if(mCurrentFrame.mvpMapPoints[i])
        {
            // 如果这个mappoint在上次优化中被标记为outlier，则剔除
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }

    return nmatchesMap>=10;
}


/**
 * @brief
 * 1.更新最近一帧的位姿
 * 2.对于双目或rgbd摄像头，为上一帧**临时生成新的MapPoints**,注意这些MapPoints不加入到Map中，在tracking的最后会删除,
 *   跟踪过程中需要将上一帧的MapPoints投影到当前帧可以缩小匹配范围，加快当前帧与上一帧进行特征点匹配
 * 
 * update mlpTemporalPoints
 */
void Tracking::UpdateLastFrame()
{
    // Update pose according to reference keyframe
    // 步骤1：更新最近一帧的位姿
    KeyFrame* pRef = mLastFrame.mpReferenceKF;
    cv::Mat Tlr = mlRelativeFramePoses.back();

    mLastFrame.SetPose(Tlr*pRef->GetPose());    // Tlr*Trw = Tlw l:last r:reference w:world

    // 如果上一帧为关键帧，或者单目的情况，则退出
    if(mnLastKeyFrameId==mLastFrame.mnId || mSensor==System::MONOCULAR || !mbOnlyTracking)
        return;

    // 步骤2：对于双目或rgbd摄像头，为上一帧临时生成新的MapPoints
    // 注意这些MapPoints不加入到Map中，在tracking的最后会删除
    // 跟踪过程中需要将上一帧的MapPoints投影到当前帧可以缩小匹配范围，加快当前帧与上一帧进行特征点匹配
    // Create "visual odometry" MapPoints
    // We sort points according to their measured depth by the stereo/RGB-D sensor
    // 步骤2.1：得到上一帧有深度值的特征点

    vector<pair<float,int> > vDepthIdx; 
    vDepthIdx.reserve(mLastFrame.N);
    for(int i=0; i<mLastFrame.N;i++)
    {
        float z = mLastFrame.mvDepth[i];
        if(z>0) //如果深度大于0
        {
            vDepthIdx.push_back(make_pair(z,i));    // (depth, idx of MP)
        }
    }

    if(vDepthIdx.empty())
        return;

    // 步骤2.2：按照深度从小到大排序
    sort(vDepthIdx.begin(),vDepthIdx.end());

    // We insert all close points (depth<mThDepth)
    // If less than 100 close points, we insert the 100 closest ones.
    // 步骤2.3：将距离比较近的点包装成MapPoints
    int nPoints = 0;
    for(size_t j=0; j<vDepthIdx.size();j++)
    {
        int i = vDepthIdx[j].second;    // idx of KP

        bool bCreateNew = false;

        MapPoint* pMP = mLastFrame.mvpMapPoints[i]; // MP
        if(!pMP)
            bCreateNew = true;
        else if(pMP->Observations()<1)
        {
            bCreateNew = true;
        }

        if(bCreateNew)
        {
            // 这些生成MapPoints后并没有通过：
            // a.AddMapPoint、
            // b.AddObservation、
            // c.ComputeDistinctiveDescriptors、
            // d.UpdateNormalAndDepth添加属性，
            // 这些MapPoint仅仅为了提高双目和RGBD的跟踪成功率
            cv::Mat x3D = mLastFrame.UnprojectStereo(i);
            MapPoint* pNewMP = new MapPoint(x3D,mpMap,&mLastFrame,i);

            mLastFrame.mvpMapPoints[i]=pNewMP;

            // 标记为临时添加的MapPoint，之后在CreateNewKeyFrame之前会全部删除
            mlpTemporalPoints.push_back(pNewMP);
            nPoints++;
        }
        else
        {
            nPoints++;
        }

        if(vDepthIdx[j].first>mThDepth && nPoints>100)
            break;
    }
}


/**
 * @brief 这个函数前提假设是相机是**匀速运动**的
 * 步骤1：  首先根据**上一帧的位姿**和**上一帧相机运动的速度**来估计cur frame的**当前位姿TCW**（相机运动速度暂时不知道怎么求的）
 * 步骤2：  将上一帧带有的**mappoints**，根据步骤1估计的R，T投影到cur frame上
 * 步骤3：  设定一个阈值，在cur frame投影点附近进行搜索，以找到与上一帧中**匹配**的ORB特征点
 * 步骤4：  步骤3已经将两帧的ORB特征点进行了**匹配**，接着通过**优化3D-2D的重投影误差(PnP)**来获得cur frame的**位姿Tcw**
 * 步骤5：  优化位姿后,剔除outlier的mvpMapPoints
 * 
 * 1. 非单目情况，需要对上一帧产生一些新的MapPoints（临时）
 * 2. 将上一帧的MapPoints投影到当前帧的图像平面上，在投影的位置进行区域匹配
 * 3. 根据匹配优化当前帧的姿态
 * 4. 根据姿态剔除误匹配
 * 
 * @return 如果匹配数大于10，返回true
 */
bool Tracking::TrackWithMotionModel()
{
    ORBmatcher matcher(0.9,true);

    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points if in Localization Mode
    // 对于双目或rgbd摄像头，根据深度值为上一关键帧生成新的MapPoints
    // （跟踪过程中需要将当前帧与上一帧进行特征点匹配，将上一帧的MapPoints投影到当前帧可以缩小匹配范围） ++++
    // 在跟踪过程中，去除outlier的MapPoint，如果不及时增加MapPoint会逐渐减少
    // 这个函数的功能就是补充增加RGBD和双目相机上一帧的MapPoints数
    UpdateLastFrame();

    // 将 当前帧的初始位姿 设为上一帧位姿乘上一帧位姿的变化速度，得到当前的 R，T
    mCurrentFrame.SetPose(mVelocity*mLastFrame.mTcw);

    fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));

    // Project points seen in previous frame
    int th;
    if(mSensor!=System::STEREO)
        th=15;
    else
        th=7;
    
    // 对上一帧的MapPoints进行跟踪，看上一帧能看到的mappoint对应的当前帧哪些特征点
    // 根据上一帧特征点对应的3D点投影的位置**缩小特征点匹配范围**
    int nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR);

    // If few matches, uses a wider window search
    // 匹配数量太少，扩大特征匹配搜索框重新进行mappoint跟踪
    if(nmatches<20)
    {
        fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
        nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,2*th,mSensor==System::MONOCULAR);
    }

    if(nmatches<20)
        return false;

    // Optimize frame pose with all matches
    // 优化位姿
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    // 上一步的位姿优化更新了mCurrentFrame的outlier，需要将mCurrentFrame的**mvpMapPoints**更新
    // mvpMapPoints:    MapPoints associated to keypoints, NULL pointer if no association
    // mvbOutlier:      Flag to identify outlier associations
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)   // N: Number of KeyPoints
    {
        if(mCurrentFrame.mvpMapPoints[i])   // has a MapPoint
        {
            if(mCurrentFrame.mvbOutlier[i]) // is Outlier
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            // 如果当前帧可以看到的mappoint同时能被其他keyframe看到
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }    

    if(mbOnlyTracking)
    {
        mbVO = nmatchesMap<10;
        return nmatches>20;
    }

    return nmatchesMap>=10; // 匹配数
}


/**
 * @brief 对mvpLocalKeyFrames，mvpLocalMapPoints进行跟踪
 *
 * 投影，从已经生成的地图点中找到更多对应关系
 * 1.更新Covisibility Graph， 更新局部关键帧
 * 2.根据局部关键帧，更新局部地图点，接下来运行过滤函数 isInFrustum
 * 3.将地图点投影到当前帧上，超出图像范围的舍弃
 * 4.当前视线方向v和地图点云平均视线方向n, 舍弃n*v<cos(60)的点云
 * 5.舍弃地图点到相机中心距离不在一定阈值内的点
 * 6.计算图像的尺度因子 isInFrustum 函数结束
 * 7.进行非线性最小二乘优化
 * 8.更新地图点的统计量
 * 
 * 1. 更新局部地图，包括局部KF和KP
 * 2. 以局部地图的mappoint为范围和当前帧进行特征匹配
 * 3. 根据匹配对通过BA估计**当前帧的姿态**
 * 4. 更新当前帧的MapPoints被观测程度，并统计跟踪局部地图的效果
 * @return 根据跟踪局部地图的效果判断当前帧的跟踪成功与否，返回其判断结果
 * @see V-D track Local Map
 */
bool Tracking::TrackLocalMap()
{
    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.
    // 更新局部地图，即更新局部地图关键帧，局部地图mappoint
    // keyframe, mappoint
    UpdateLocalMap();

    // 在局部地图的mappoint中查找在当前帧视野范围内的点，将视野范围内的点和当前帧的特征点进行投影匹配
    // keypoint, ? which point?
    SearchLocalPoints();

    // 在这个函数之前，在Relocalization、TrackReferenceKeyFrame、TrackWithMotionModel中都有位姿优化，
    // 更新局部所有MapPoints后对位姿再次优化
    // update (MP) -> optimize (Pose)
    Optimizer::PoseOptimization(&mCurrentFrame);
    mnMatchesInliers = 0;   // 标记该mappoint点被当前帧观测, match a keypoint

    // Update MapPoints Statistics
    // 更新当前帧的MapPoints被**观测程度**，并统计跟踪局部地图的效果
    // 观测程度: 1) the keypoint corresponds to a MapPoint 2) keypoint is not a outlier
    // what is the criteria of outlier?
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(!mCurrentFrame.mvbOutlier[i])
            {
                // 标记该mappoint点被当前帧观测->inlier
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                if(!mbOnlyTracking)
                {
                    if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                        mnMatchesInliers++; // update
                }
                else
                    mnMatchesInliers++;     // update
            }
            else if(mSensor==System::STEREO)
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);

        }
    }

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    // 决定是否跟踪成功
    // 如果当前帧和上一次重定位太近
    // 或者当前帧特征点与mappoint的匹配数太少
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && mnMatchesInliers<50)
        return false;

    if(mnMatchesInliers<30)
        return false;
    else
        return true;
}


/**
 * 判断是否需要添加新的keyframe
 * 
 * Tracking线程需要在尾部决定是否创建新的关键帧，本函数就完成了此判断。
 * 本函数的判断结果主要取决于当前跟踪效果是否理想、当前帧的跟踪质量好坏、当前局部建图线程是否有空。
 * 如果Tracking处于只跟踪模式，则本函数直接返回false，不需要新关键帧；
 * 如果局部建图线程因为检测到了闭环而被终止或正在终止，则本函数直接返回false；
 * 如果距离上一次重定位不久，直接返回false。
 * 
 * 判断是否需要生成新的关键帧标准
 * （1）在上一个全局重定位后，又过了20帧；
 * （2）局部建图闲置，或在上一个关键帧插入后，又过了20帧；
 * （3）当前帧跟踪到大于50个点；
 * （4）当前帧跟踪到的比参考关键帧少90%
 */
bool Tracking::NeedNewKeyFrame()
{
    // 步骤1：如果用户在界面上选择重定位，那么将不插入关键帧
    // 由于插入关键帧过程中会生成MapPoint，因此用户选择重定位后地图上的点云和关键帧都不会再增加
    if(mbOnlyTracking) // 如果仅跟踪，不选关键帧
        return false;

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    // 如果局部地图被闭环检测使用，则不插入关键帧
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
        return false;

    const int nKFs = mpMap->KeyFramesInMap();

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    // 步骤2：判断是否距离上一次插入关键帧的时间太短
    // mCurrentFrame.mnId 是当前帧的ID
    // mnLastRelocFrameId 是最近一次重定位帧的ID
    // mMaxFrames 等于图像输入的帧率
    // 如果关键帧比较少，则考虑插入关键帧
    // 或距离上一次重定位超过1s，则考虑插入关键帧
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && nKFs>mMaxFrames)
        return false;

    // Tracked MapPoints in the reference keyframe
    // 步骤3：得到参考关键帧跟踪到的MapPoints数量
    // 在 UpdateLocalKeyFrames 函数中会将**与当前关键帧共视程度最高**的关键帧设定为当前帧的参考关键帧
    int nMinObs = 3;
    if(nKFs<=2)
        nMinObs=2;
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

    // Local Mapping accept keyframes?
    // 步骤4：查询局部地图管理器是否繁忙
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    // Check how many "close" points are being tracked and how many could be potentially created.
    int nNonTrackedClose = 0;
    int nTrackedClose= 0;
    if(mSensor!=System::MONOCULAR) // 双目或rgbd
    {
        for(int i =0; i<mCurrentFrame.N; i++)
        {
            if(mCurrentFrame.mvDepth[i]>0 && mCurrentFrame.mvDepth[i]<mThDepth)
            {
                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                    nTrackedClose++;
                else
                    nNonTrackedClose++;
            }
        }
    }

    bool bNeedToInsertClose = (nTrackedClose<100) && (nNonTrackedClose>70);

    // Thresholds
    // 设定inlier阈值，和之前帧特征点匹配的inlier比例
    float thRefRatio = 0.75f;
    if(nKFs<2)
        thRefRatio = 0.4f;

    if(mSensor==System::MONOCULAR)
        thRefRatio = 0.9f;

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    // 和上一个关键帧间隔需要大于 mMaxFrames
    const bool c1a = mCurrentFrame.mnId>=mnLastKeyFrameId+mMaxFrames;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    // 如果Local Mapping空闲，且和上一个关键帧间隔需要大于 mMinFrames
    const bool c1b = (mCurrentFrame.mnId>=mnLastKeyFrameId+mMinFrames && bLocalMappingIdle);
    // Condition 1c: tracking is weak
    const bool c1c =  mSensor!=System::MONOCULAR && (mnMatchesInliers<nRefMatches*0.25 || bNeedToInsertClose) ;
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    const bool c2 = ((mnMatchesInliers<nRefMatches*thRefRatio|| bNeedToInsertClose) && mnMatchesInliers>15);

    if((c1a||c1b||c1c)&&c2)
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        if(bLocalMappingIdle)
        {
            return true;
        }
        else
        {
            mpLocalMapper->InterruptBA();
            if(mSensor!=System::MONOCULAR)
            {
                if(mpLocalMapper->KeyframesInQueue()<3)
                    return true;
                else
                    return false;
            }
            else
                return false;
        }
    }
    else
        return false;
}


/**
 * 创建新的关键帧，对于非单目的情况，同时创建新的 MapPoints
 */
void Tracking::CreateNewKeyFrame()
{
    if(!mpLocalMapper->SetNotStop(true))
        return;

    // 步骤1：将当前帧构造成关键帧
    KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

    // 步骤2：将当前关键帧设置为**当前帧的参考关键帧**
    // 在 UpdateLocalKeyFrames 函数中会将**与当前关键帧**共视程度最高的关键帧**设定为**当前帧的参考关键帧**
    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;

    // 这段代码和UpdateLastFrame中的那一部分代码功能相同
    // 步骤3：对于双目或rgbd摄像头，为当前帧生成新的MapPoints
    if(mSensor!=System::MONOCULAR)
    {
        // 根据Tcw计算mRcw、mtcw和mRwc、mOw
        mCurrentFrame.UpdatePoseMatrices();

        // We sort points by the measured depth by the stereo/RGBD sensor.
        // We create all those MapPoints whose depth < mThDepth.
        // If there are less than 100 close points we create the 100 closest.
        // 步骤3.1：得到当前帧深度小于阈值的特征点(keypoint)
        // 创建新的MapPoint, depth < mThDepth
        vector<pair<float,int> > vDepthIdx;
        vDepthIdx.reserve(mCurrentFrame.N);
        for(int i=0; i<mCurrentFrame.N; i++)            // N: keypoints
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                vDepthIdx.push_back(make_pair(z,i));    // (depth, idx of keypoint)
            }
        }

        if(!vDepthIdx.empty())
        {
            // 步骤3.2：按照深度**从小到大排序**
            sort(vDepthIdx.begin(),vDepthIdx.end());

            // 步骤3.3：将**距离比较近的**点包装成MapPoints
            int nPoints = 0;
            for(size_t j=0; j<vDepthIdx.size();j++)
            {
                int i = vDepthIdx[j].second;                    // idx of keypoint

                bool bCreateNew = false;

                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];  // MapPoint
                if(!pMP)
                    bCreateNew = true;
                else if(pMP->Observations()<1)
                {
                    bCreateNew = true;
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                }

                if(bCreateNew)
                {
                    cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                    MapPoint* pNewMP = new MapPoint(x3D,pKF,mpMap);
                    // 这些添加属性的操作是每次创建MapPoint后都要做的
                    pNewMP->AddObservation(pKF,i);
                    pKF->AddMapPoint(pNewMP,i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    // 向map中添加mappoint
                    mpMap->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                    nPoints++;
                }
                else
                {
                    nPoints++;
                }
                // 如果深度大于阈值mThDepth，或者nPoints>100则停止添加mappoint点
                // 这里决定了双目和rgbd摄像头时地图点云的稠密程度
                // 但是仅仅为了让地图稠密直接改这些不太好，
                // 因为这些MapPoints会参与之后整个slam过程
                if(vDepthIdx[j].first>mThDepth && nPoints>100)
                    break;
            }
        }
    }

    mpLocalMapper->InsertKeyFrame(pKF);

    mpLocalMapper->SetNotStop(false);

    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF;
}


/**
 * 在局部地图的MapPoint中查找在当前帧视野范围内的点(MapPoint)
 * 将视野范围内的点(MapPoint)和当前帧的特征点进行投影匹配
 */
void Tracking::SearchLocalPoints()
{
    // Do not search map points already matched
    // 当前帧mCurrentFrame匹配的mappoint点就不要匹配了
    // 这些匹配点都是在***TrackWithMotionModel()***，***TrackReferenceKeyFrame()***，***Relocalization()***中当前帧和mappoint的匹配
    // 它们都是被“预测”和当前帧匹配的mappoint点匹配
    // Frame::mvpMapPoints 该容器被更新 ， 这里没有在匹配上的MapPoint中添加该帧的观测
    //
    // 步骤1：遍历当前帧的mvpMapPoints，标记这些MapPoints不参与之后的搜索
    // 因为当前的mvpMapPoints一定在当前帧的视野中
    for(vector<MapPoint*>::iterator vit=mCurrentFrame.mvpMapPoints.begin(), vend=mCurrentFrame.mvpMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP)
        {
            if(pMP->isBad())
            {
                *vit = static_cast<MapPoint*>(NULL);
            }
            else
            {
                // 标记这些MapPoints不参与之后的搜索
                // 预测这个mappoint会被匹配
                pMP->IncreaseVisible();
                // 标记该点被当前帧观测到
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                // 标记该点将来在matcher.SearchByProjection()不被投影，因为已经匹配过
                pMP->mbTrackInView = false;
            }
        }
    }

    int nToMatch=0;

    // Project points in frame and check its visibility
    // 遍历刚才更新的 mvpLocalMapPoints ，筛选哪些不在视野范围内的mappoint
    // 在视野范围内的mappoint是被预测我们能够和当前帧匹配上的mappoint点
    //
    // 步骤2：将所有局部MapPoints投影到当前帧，判断是否在视野范围内，然后进行投影匹配
    for(vector<MapPoint*>::iterator vit=mvpLocalMapPoints.begin(), vend=mvpLocalMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        // 已经被当前帧观测到MapPoint不再判断是否能被当前帧观测到
        if(pMP->mnLastFrameSeen == mCurrentFrame.mnId)
            continue;
        if(pMP->isBad())
            continue;
        // Project (this fills MapPoint variables for matching)
        // 如果此mappoint点**在视野范围内**
        // Frame::isInFrustum() 判断局部地图中是否有在当前帧视野内的MapPoint点
        if(mCurrentFrame.isInFrustum(pMP,0.5))
        {
            // 预测这个mappoint会被匹配
            // 观测到该点的帧数加1，该MapPoint在某些帧的视野范围内
            pMP->IncreaseVisible();
            // 只有在视野范围内的MapPoints才参与之后的投影匹配
            nToMatch++;
        }
    }

    if(nToMatch>0)
    {
        ORBmatcher matcher(0.8);
        int th = 1;
        if(mSensor==System::RGBD)
            th=3;
        // If the camera has been relocalised recently, perform a coarser search
        // 如果不久前进行过重定位，那么进行一个更加宽泛的搜索，阈值需要增大
        if(mCurrentFrame.mnId<mnLastRelocFrameId+2)
            th=5;
        // ORBmatcher::SearchByProjection() 将局部地图中的MapPoint与当前帧做投影匹配
        // 步骤2.2：对视野范围内的MapPoints通过投影进行特征点匹配
        matcher.SearchByProjection(mCurrentFrame,mvpLocalMapPoints,th);
    }
}

void Tracking::UpdateLocalMap()
{
    // This is for visualization
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    // Update
    UpdateLocalKeyFrames();
    UpdateLocalPoints();
}


/**
 * @brief 将mvpLocalKeyFrames中的mappoint，添加到局部地图关键点mvpLocalMapPoints中
 */
void Tracking::UpdateLocalPoints()
{
    // 清空局部地图关键点
    mvpLocalMapPoints.clear();

    // for each KeyFrames
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        KeyFrame* pKF = *itKF;
        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

        for(vector<MapPoint*>::const_iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++)
        {
            MapPoint* pMP = *itMP;
            if(!pMP)
                continue;

            // mnTrackReferenceForFrame 防止重复添加局部MapPoint
            if(pMP->mnTrackReferenceForFrame==mCurrentFrame.mnId)
                continue;
            if(!pMP->isBad())
            {
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame=mCurrentFrame.mnId;
            }
        }
    }
}


/**
 * @brief 更新 mpReferenceKF ，mCurrentFrame.mpReferenceKF
 * 更新局部地图关键帧 mvpLocalKeyFrames
 */
void Tracking::UpdateLocalKeyFrames()
{
    // Each map point vote for the keyframes in which it has been observed
    // 遍历当前帧的mappoint，将所有能观测到这些mappoint的keyframe，及其可以观测的这些mappoint数量存入keyframeCounter
    map<KeyFrame*,int> keyframeCounter; // <Keyframe, number of MPs>
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if(!pMP->isBad())
            {
                const map<KeyFrame*,size_t> observations = pMP->GetObservations();
                for(map<KeyFrame*,size_t>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                    keyframeCounter[it->first]++;
            }
            else
            {
                mCurrentFrame.mvpMapPoints[i]=NULL;
            }
        }
    }

    if(keyframeCounter.empty())
        return;

    int max=0;
    KeyFrame* pKFmax= static_cast<KeyFrame*>(NULL);

    // 先清空局部地图关键帧
    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3*keyframeCounter.size());

    // **All** keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    // 向 mvpLocalKeyFrames 添加能观测到当前帧MapPoints的关键帧
    for(map<KeyFrame*,int>::const_iterator it=keyframeCounter.begin(), itEnd=keyframeCounter.end(); it!=itEnd; it++)
    {
        KeyFrame* pKF = it->first;

        if(pKF->isBad())
            continue;

        // 更新max，pKFmax，以寻找**能看到最多mappoint的keyframe**
        if(it->second>max)
        {
            max=it->second;
            pKFmax=pKF;
        }

        mvpLocalKeyFrames.push_back(it->first);
        // mnTrackReferenceForFrame 防止重复添加局部地图关键帧
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }


    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    // 遍历 mvpLocalKeyFrames ，以向 mvpLocalKeyFrames 添加更多的关键帧。有三种途径：
    // 1.取出此关键帧itKF在Covisibilitygraph中共视程度最高的10个关键帧；
    // 2.取出此关键帧itKF在Spanning tree中的子节点；
    // 3.取出此关键帧itKF在Spanning tree中的父节点；
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if(mvpLocalKeyFrames.size()>80)
            break;

        KeyFrame* pKF = *itKF;

        // 1.取出此关键帧itKF在essential graph中共视程度最高的10个关键帧
        const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

        for(vector<KeyFrame*>::const_iterator itNeighKF=vNeighs.begin(), itEndNeighKF=vNeighs.end(); itNeighKF!=itEndNeighKF; itNeighKF++)
        {
            KeyFrame* pNeighKF = *itNeighKF;
            if(!pNeighKF->isBad())
            {
                if(pNeighKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    // 向 mvpLocalKeyFrames 添加更多的关键帧
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        // 2.取出此关键帧itKF在Spanning tree中的 子节点
        // Spanning tree的节点为关键帧，共视程度最高的那个关键帧设置为节点在Spanning Tree中的父节点
        const set<KeyFrame*> spChilds = pKF->GetChilds();
        for(set<KeyFrame*>::const_iterator sit=spChilds.begin(), send=spChilds.end(); sit!=send; sit++)
        {
            KeyFrame* pChildKF = *sit;
            if(!pChildKF->isBad())
            {
                if(pChildKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    // 向 mvpLocalKeyFrames 添加更多的关键帧
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        // 3.取出此关键帧itKF在Spanning tree中的父节点
        KeyFrame* pParent = pKF->GetParent();
        if(pParent)
        {
            if(pParent->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                break;
            }
        }

    }

    if(pKFmax)
    {
        // 更新参考关键帧为有共视的mappoint关键帧共视程度最高（共视的mappoint数量最多）的关键帧
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}


/**
 * 作用：重定位，从之前的关键帧中找出 与当前帧之间拥有充足匹配点的 候选帧 ，利用Ransac迭代，通过PnP求解位姿
 * 
 * 1）先计算当前帧的BOW值，并从 关键帧数据库 中查找**候选的匹配candidates**
 * 2）构建 PnP求解器 ，标记杂点，准备好 每个关键帧和当前帧的 匹配点集
 * 3）用PnP算法求解位姿，进行若干次P4P Ransac迭代，并使用非线性最小二乘优化，直到发现一个有充足 inliers 支持的 相机位置
 * 4）返回成功或失败
 * 
 * 需要利用 relocalization 到 全局的地图中 查找匹配帧，计算位姿
 * 
 * 假如当前帧与最近邻关键帧的匹配也失败了，那么意味着此时当前帧已经丢了，无法确定其真实位置。
 * 此时，只有去和所有关键帧匹配，看能否找到合适的位置。
 * 首先，利用BoW词典选取若干关键帧作为备选（参见ORB－SLAM（六）回环检测）
 * 其次，寻找有足够多的特征点匹配的关键帧
 * 最后，利用特征点匹配迭代求解位姿（RANSAC框架下，因为相对位姿可能比较大，局外点会比较多）。
 * 如果有关键帧有足够多的内点，那么选取该关键帧优化出的 位姿
 * 
 * @return 直到发现一个有充足 inliers 支持的 相机位置 for this F
 */
bool Tracking::Relocalization()
{
    // Compute Bag of Words Vector
    // 步骤1：计算当前帧特征点的Bow映射
    mCurrentFrame.ComputeBoW();

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    // Detect Relocalization Candidates/KeyFrames
    // 步骤2：找到 与当前帧相似的 候选关键帧
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);

    // 如果候选关键帧为空，则返回Relocalization失败
    if(vpCandidateKFs.empty())
        return false;

    const int nKFs = vpCandidateKFs.size(); // 候选关键帧个数

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    // 首先执行 与每个候选匹配的 ORB匹配
    // 如果找到足够的匹配，设置一个**PNP解算器**->Pose
    ORBmatcher matcher(0.75,true);

    vector<PnPsolver*> vpPnPsolvers;
    vpPnPsolvers.resize(nKFs);

    // 表示各个候选帧的**mappoint**与和当前帧**特征点**的匹配
    // 现在你想把mCurrentFrame的特征点和mappoint进行匹配，有个便捷的方法就是，
    // KeyPoint    <->  KeyPoint  <->MapPoint
    // mCurrentFrame|candidateFrame|candidateFrame
    // 让mCurrentFrame特征点和候选关键帧的特征点进行匹配,然后我们是知道候选关键帧特征点与mappoint的匹配的
    // 这样就能够将mCurrentFrame特征点和mappoint匹配起来了，相当于通过和候选关键帧这个桥梁匹配上了mappoint
    // vvpMapPointMatches[i][j]就表示mCurrentFrame的第j个特征点如果是经由第i个候选关键帧匹配mappoint，是哪个mappoint
    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    int nCandidates=0;

    // 候选帧和当前帧进行特征匹配，剔除匹配数量少的候选关键帧
    // 为未被剔除的关键帧就新建PnPsolver，准备在后面进行epnp
    // for each candidate
    for(int i=0; i<nKFs; i++)
    {
        // candidate KF
        KeyFrame* pKF = vpCandidateKFs[i];
        if(pKF->isBad())
            vbDiscarded[i] = true; // 去除不好的候选关键帧
        else
        {
            // 步骤3：通过BoW进行匹配
            // mCurrentFrame与候选关键帧进行**特征点匹配**
            // KeyPoint    <->  KeyPoint 
            // mCurrentFrame|candidateFrame
            int nmatches = matcher.SearchByBoW(pKF,mCurrentFrame,vvpMapPointMatches[i]);
            if(nmatches<15) // 如果匹配点小于15剔除
            {
                vbDiscarded[i] = true;
                continue;
            }
            else // 用pnp求解
            {
                PnPsolver* pSolver = new PnPsolver(mCurrentFrame,vvpMapPointMatches[i]);
                pSolver->SetRansacParameters(0.99,10,300,4,0.5,5.991);
                vpPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    bool bMatch = false;
    ORBmatcher matcher2(0.9,true);

    // 大概步骤是这样的，小循环for不断的遍历剩下的nCandidates个的候选帧，这些候选帧对应有各自的PnPsolvers
    // 第i次for循环所对应的vpPnPsolvers[i]就会执行5次RANSAC循环求解出**5个位姿**
    // 通过计算5个位姿对应的匹配点的inliner数量来判断位姿的好坏。如果这5个位姿比记录中的最好位姿更好，更新**最好位姿**以及对应的匹配点哪些点是inliner
    // 如果最好的那个位姿inliner超过阈值，或者vpPnPsolvers[i]RANSAC累计迭代次数超过阈值，都会把位姿拷贝给Tcw。否则Tcw为空
    // 如果Tcw为空，那么就循环计算下一个vpPnPsolvers[i+1]
    // 通过5次RANSAC求解位姿后，如果Tcw不为空，这继续判断它是否和当前帧匹配
    while(nCandidates>0 && !bMatch)
    {
        // For each candidate, get the **pose**
        // nKFs: 候选关键帧个数
        for(int i=0; i<nKFs; i++)
        {
            if(vbDiscarded[i])
                continue;

            // Perform 5 Ransac Iterations
            // 此次RANSAC会计算出一个位姿，在这个位姿下，mCurrentFrame中的特征点哪些是有mappoint匹配的
            // **inliner**=> keyPoint <-> MapPoint
	        // vbInliers大小是mCurrentFrame中的特征点数量大小
            vector<bool> vbInliers;
            int nInliers;   // vbInliers大小
            bool bNoMore;

            // 步骤4：通过EPnP算法 估计姿态 Tcw
            PnPsolver* pSolver = vpPnPsolvers[i];
            // 通过EPnP算法估计姿态，有5次RANSAC循环 => Tcw
            cv::Mat Tcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers);

            // If Ransac reachs max. iterations discard keyframe
            // 如果RANSAC循环达到了最大
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            // 相机姿态算出来有两种情况，一种是在RANSAC累计迭代次数没有达到mRansacMaxIts之前，找到了一个复合要求的**位姿**
            // 另一种情况是RANSAC累计迭代次数到达了最大mRansacMaxIts
            // 如果相机姿态已经算出来了，优化它
            if(!Tcw.empty())
            {
                Tcw.copyTo(mCurrentFrame.mTcw);

                set<MapPoint*> sFound;

                // np为mCurrentFrame的特征点数量
                const int np = vbInliers.size(); // 内点个数

                // 根据vbInliers更新mCurrentFrame.mvpMapPoints
                // 也就是根据vbInliers更新mCurrentFrame的特征点与哪些mappoint匹配
                // set<MapPoint*> sFound;
		        // 并记下当前mCurrentFrame与哪些mappoint匹配到 sFound ，以便后面快速查询
                // vvpMapPointMatches[i][j]就表示mCurrentFrame的第j个特征点如果是经由第i个候选关键帧匹配mappoint，是哪个mappoint
                for(int j=0; j<np; j++)
                {
                    if(vbInliers[j])
                    {
                        mCurrentFrame.mvpMapPoints[j]=vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        mCurrentFrame.mvpMapPoints[j]=NULL;
                }

                // 步骤5：通过 PoseOptimization 对姿态进行优化求解
                // BA优化位姿
                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                if(nGood<10)
                    continue;

                // 剔除PoseOptimization算出的mvbOutlier
                for(int io =0; io<mCurrentFrame.N; io++)
                    if(mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io]=static_cast<MapPoint*>(NULL);

                // If few inliers, search by projection in a coarse window and optimize again
                // 步骤6：如果内点较少，则 通过投影的方式对之前未匹配的点进行 匹配，再进行优化求解
                if(nGood<50)
                {
                    // mCurrentFrame中特征点已经匹配好一些mappoint在sFound中，如果内点较少,mCurrentFrame想要更多的mappoint匹配
                    // 于是通过matcher2.SearchByProjection函数将vpCandidateKFs[i]的mappoint悉数投影到CurrentFrame再就近搜索特征点进行匹配
		            // mCurrentFrame返回得到通过此函数匹配到的新mappoint的个数
                    int nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,10,100);

                    // 如果目前mCurrentFrame匹配到的mappoint个数超过50
                    if(nadditional+nGood>=50)
                    {
                        // 优化位姿，返回(nadditional+nGood)有多少点是inliner
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        // 如果nGood不够多，那缩小搜索框重复再匹配一次
                        if(nGood>30 && nGood<50)
                        {
                            // set<MapPoint*> sFound;
		                    // 更新sFound，并记下当前mCurrentFrame与哪些mappoint匹配到 sFound
                            sFound.clear();
                            for(int ip =0; ip<mCurrentFrame.N; ip++)
                                if(mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);

                            // 缩小搜索框重复再匹配一次,返回这个新得到的匹配数
                            nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,3,64);

                            // Final optimization
                            if(nGood+nadditional>=50)
                            {
                                // 位姿优化，可能又会有些匹配消失
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                // 将刚才位姿优化得到的outliner匹配更新
                                for(int io =0; io<mCurrentFrame.N; io++)
                                    if(mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io]=NULL;
                            }
                        }
                    }
                }


                // If the pose is supported by enough inliers stop ransacs and continue
                if(nGood>=50)
                {
                    bMatch = true;
                    break;
                }
            }
        }
    }

    if(!bMatch)
    {
        return false;
    }
    else
    {
        mnLastRelocFrameId = mCurrentFrame.mnId;
        return true;
    }

}


void Tracking::Reset()
{

    cout << "System Reseting" << endl;
    if(mpViewer)
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped())
            usleep(3000);
    }

    // Reset Local Mapping
    cout << "Reseting Local Mapper...";
    mpLocalMapper->RequestReset();
    cout << " done" << endl;

    // Reset Loop Closing
    cout << "Reseting Loop Closing...";
    mpLoopClosing->RequestReset();
    cout << " done" << endl;

    // Clear BoW Database
    cout << "Reseting Database...";
    mpKeyFrameDB->clear();
    cout << " done" << endl;

    // Clear Map (this erase MapPoints and KeyFrames)
    mpMap->clear();

    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;

    if(mpInitializer)
    {
        delete mpInitializer;
        mpInitializer = static_cast<Initializer*>(NULL);
    }

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();

    if(mpViewer)
        mpViewer->Release();
}

void Tracking::ChangeCalibration(const string &strSettingPath)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    Frame::mbInitialComputations = true;
}

void Tracking::InformOnlyTracking(const bool &flag)
{
    mbOnlyTracking = flag;
}



} //namespace ORB_SLAM
