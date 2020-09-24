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

#ifndef FRAME_H
#define FRAME_H

#include<vector>

#include "MapPoint.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include "ORBVocabulary.h"
#include "KeyFrame.h"
#include "ORBextractor.h"

#include <opencv2/opencv.hpp>

namespace ORB_SLAM2
{
#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64

class MapPoint;
class KeyFrame;

class Frame
{
public:
    Frame();

    // Copy constructor.
    Frame(const Frame &frame);

    // Constructor for stereo cameras.
    Frame(const cv::Mat &imLeft, const cv::Mat &imRight, 
            const double &timeStamp, 
            ORBextractor* extractorLeft, ORBextractor* extractorRight, 
            ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, 
            const float &bf, const float &thDepth);

    // Constructor for RGB-D cameras.
    Frame(const cv::Mat &imGray, const cv::Mat &imDepth, 
            const double &timeStamp, 
            ORBextractor* extractor,
            ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, 
            const float &bf, const float &thDepth);

    // Constructor for Monocular cameras.
    Frame(const cv::Mat &imGray, 
            const double &timeStamp, 
            ORBextractor* extractor,
            ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, 
            const float &bf, const float &thDepth);

    // Extract ORB on the image. 0 for left image and 1 for right image.
    void ExtractORB(int flag, const cv::Mat &im);

    // Compute Bag of Words representation.
    void ComputeBoW();

    // Set the camera pose.
    void SetPose(cv::Mat Tcw);

    // Computes rotation, translation and camera center matrices from the camera pose.
    void UpdatePoseMatrices();

    // Returns the camera center.
    inline cv::Mat GetCameraCenter(){
        return mOw.clone();
    }

    // Returns inverse of rotation
    inline cv::Mat GetRotationInverse(){
        return mRwc.clone();
    }


    // Check if a MapPoint is in the frustum of the camera
    // and fill variables of the MapPoint to be used by the tracking
    // pMP是否在当前帧视野范围内
    bool isInFrustum(MapPoint* pMP, float viewingCosLimit);


    // Compute the cell of a keypoint (return false if outside the grid)
    // 计算kp在哪一个窗格，如果超出边界则返回false
    bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);


    /**
     * function() const {}
     * A function becomes const when the const keyword is used in the function’s 
     * declaration. The idea of const functions is not to allow them to modify 
     * the object on which they are called. It is recommended the practice to 
     * make as many functions const as possible so that accidental changes to 
     * objects are avoided.
     * 
     * When a function is declared as const, it can be called on any type of object. 
     * Non-const functions can only be called by non-const objects.
     * 
     * non-const objects.Non-const functions
     * 
     * non-const objects.const functions
     * const objects.const functions
     * 
     * https://www.geeksforgeeks.org/const-member-functions-c/
     * 
     * 
     * 找到在 以x, y为中心,边长为2r的方形搜索框内且在[minLevel, maxLevel]的特征点
     * @param x        图像坐标u
     * @param y        图像坐标v
     * @param r        边长
     * @param minLevel 最小尺度
     * @param maxLevel 最大尺度
     * @return         满足条件的特征点的序号
     */
    vector<size_t> GetFeaturesInArea(const float &x, 
                                     const float &y, 
                                     const float &r, 
                                     const int minLevel=-1, 
                                     const int maxLevel=-1) const;


    // Search a match for each keypoint in the left image to a keypoint in the right image.
    // If there is a match, depth is computed and the right coordinate associated to the 
    // left keypoint is stored.
    void ComputeStereoMatches();

    // Associate a "right" coordinate to a keypoint if there is valid depth in the depthmap.
    void ComputeStereoFromRGBD(const cv::Mat &imDepth);

    // Backprojects a keypoint (if stereo/depth info available) into 3D world coordinates.
    cv::Mat UnprojectStereo(const int &i);

public:
    // Vocabulary used for relocalization.
    ORBVocabulary* mpORBvocabulary;

    // Feature extractor. The right is used only in the stereo case.
    ORBextractor* mpORBextractorLeft, *mpORBextractorRight;

    // Frame timestamp.
    double mTimeStamp;

    // Calibration matrix and OpenCV distortion parameters.
    cv::Mat mK;
    static float fx;
    static float fy;
    static float cx;
    static float cy;
    static float invfx;
    static float invfy;
    cv::Mat mDistCoef;

    // Stereo baseline multiplied by fx.
    float mbf;

    // Stereo baseline in meters.
    float mb;

    // Threshold close/far points. Close points are inserted from 1 view.
    // Far points are inserted as in the monocular case from 2 views.
    float mThDepth;

    // Number of KeyPoints.
    int N;


    // Vector of keypoints (original for visualization) and undistorted (actually used by the system).
    // In the stereo case, mvKeysUn is redundant as images must be rectified.
    // In the RGB-D case, RGB images can be distorted.
    std::vector<cv::KeyPoint> mvKeys, mvKeysRight;      // 畸变的orb关键点
    std::vector<cv::KeyPoint> mvKeysUn;                 // 纠正后的关键点


    // Corresponding stereo coordinate and depth for each keypoint.
    // "Monocular" keypoints have a negative value.
    std::vector<float> mvuRight;
    std::vector<float> mvDepth;


    // Bag of Words Vector structures.
    /**
     * mBowVec 本质是一个map<WordId, WordValue>
     * 对于某幅图像A，它的特征点可以对应多个单词，组成它的bow
     * 
     * 其中 BowVector 很好理解——就是用来表示图像的向量（同描述子类似）具体形式为[[在词典特征索引，权重]，[在词典特征索引，权重]，。。。]，
     * 计算两图像的相似度最本质的就是计算这个向量两两间的**dist距离**
     */
    DBoW2::BowVector mBowVec;


    /**
     * mFeatVec 是一个std::map<NodeId, std::vector<unsigned int>>
     * 将此帧的特征点分配到 mpORBVocabulary 树各个结点(node)，从而得到 mFeatVec
     * mFeatVec->first      代表结点ID
     * mFeatVec->second     std::vector<在mFeatVec->first结点的**特征点序号**>
     * 
     * Vector of nodes with indexes of local features
     * class FeatureVector: 
     *   public std::map<NodeId, std::vector<unsigned int> >
     * 
     * 这个向量是个map，其中以一张图片的**每个特征点**在词典某一层节点下**为条件进行分组，用来加速图形特征匹配
     * ----两两图像特征匹配只需要对相同 NodeId 下的特征点进行匹配就好
     */
    DBoW2::FeatureVector mFeatVec;


    // ORB descriptor, each row associated to a keypoint.
    cv::Mat mDescriptors, mDescriptorsRight;


    // MapPoints associated to keypoints, NULL pointer if no association.
    // 大小是 mvKeys(KeyPoints) 大小，表示mappoint和此帧特征点的联系。如果没有联系则为NULL
    std::vector<MapPoint*> mvpMapPoints;


    // Flag to identify outlier associations.
    // 描述比如经过位姿优化后，有哪些特征点是可以匹配上mappoint的，
    // 一般情况下他和mvpMapPoints描述的情况相同
    // 它比mvpMapPoints时效性更强
    std::vector<bool> mvbOutlier;


    // Keypoints are assigned to cells in a grid to reduce matching complexity 
    // when projecting MapPoints.
    static float mfGridElementWidthInv;     // x轴窗格宽倒数
    static float mfGridElementHeightInv;    // y轴窗格高倒数
    std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];   // 储存这各个窗格的特征点在mvKeysUn中的**序号**

    // Camera pose.
    cv::Mat mTcw;

    // Current and Next Frame id.
    static long unsigned int nNextId;       // 静态变量，Next Frame id
    long unsigned int mnId;                 // Current Frame id

    // Reference Keyframe.
    // 参考关键帧，有共视mappoint, 共视程度最高（共视的mappoint数量最多）的关键帧
    KeyFrame* mpReferenceKF;

    // Scale pyramid info.
    // 从orbextractor拷贝的关于高斯金字塔的信息
    int mnScaleLevels;                      // 高斯金字塔层数
    float mfScaleFactor;
    float mfLogScaleFactor;
    vector<float> mvScaleFactors;
    vector<float> mvInvScaleFactors;
    vector<float> mvLevelSigma2;
    vector<float> mvInvLevelSigma2;

    // Undistorted Image Bounds (computed once).
    static float mnMinX;
    static float mnMaxX;
    static float mnMinY;
    static float mnMaxY;

    static bool mbInitialComputations;


private:

    // Undistort keypoints given OpenCV distortion parameters.
    // Only for the RGB-D case. Stereo must be already rectified!
    // (called in the constructor).
    void UndistortKeyPoints();      // 关键点畸变矫正

    // Computes image bounds for the undistorted image (called in the constructor).
    void ComputeImageBounds(const cv::Mat &imLeft);

    // Assign keypoints to the grid for speed up feature matching (called in the constructor).
    void AssignFeaturesToGrid();

    // Rotation, translation and camera center
    cv::Mat mRcw; // Rotation from world to camera
    cv::Mat mtcw; // translation from world to camera
    cv::Mat mRwc; // Rotation from camera to world
    // 光心在世界坐标系中位姿
    cv::Mat mOw; //==mtwc, translation from camera to world
};

}// namespace ORB_SLAM

#endif // FRAME_H
