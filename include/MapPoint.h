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

#ifndef MAPPOINT_H
#define MAPPOINT_H

#include"KeyFrame.h"
#include"Frame.h"
#include"Map.h"

#include<opencv2/core/core.hpp>
#include<mutex>

namespace ORB_SLAM2
{

class KeyFrame;
class Map;
class Frame;

/**
 * image->points' descriptor/feature->landmarks
 * 
 * images->select keyframes->select keypoints->descriptor->point matching, multiple points for one landmark/mappoint->optimization for map+camera pose
 * 
 * landmark/mappoint in 3D space, containing keypoints and generic points
 * information including:
 * position: x, y, z
 * descriptor
 */
class MapPoint
{
public:
    /**
     * A landmark will be observed in several keyframes
     * In ORB SLAM, use MapPoint class to compute Covisibility (local BA) 
     * 
     * @param Pos 3D position
     * @param pRefKF reference keyframes
     * @param pFrame generic frames
     * @param pMap map
     */
    MapPoint(const cv::Mat &Pos, KeyFrame* pRefKF, Map* pMap); // for keyframes
    MapPoint(const cv::Mat &Pos,  Map* pMap, Frame* pFrame, const int &idxF); // for generic frames

    void SetWorldPos(const cv::Mat &Pos);
    cv::Mat GetWorldPos();

    cv::Mat GetNormal();
    KeyFrame* GetReferenceKeyFrame();

    std::map<KeyFrame*,size_t> GetObservations();
    int Observations();

    /**
     * add more observation information for 3D keypoints to make it be more accurate
     * the information is provided by added keyframe which includes 2D information (x,y)
     * 
     * @param pKF 
     * @param idx index of the landmark in the keyframe
     */
    void AddObservation(KeyFrame* pKF,size_t idx);

    /**
     * delete observation information
     */
    void EraseObservation(KeyFrame* pKF);


    int GetIndexInKeyFrame(KeyFrame* pKF);
    bool IsInKeyFrame(KeyFrame* pKF);

    /**
     * delete mappoint
     */
    void SetBadFlag();
    bool isBad();

    /**
     * replace mappoint
     */
    void Replace(MapPoint* pMP);    
    MapPoint* GetReplaced();

    void IncreaseVisible(int n=1);
    void IncreaseFound(int n=1);
    float GetFoundRatio();
    inline int GetFound(){
        return mnFound;
    }

    /**
     * generate descriptor
     */
    void ComputeDistinctiveDescriptors();

    cv::Mat GetDescriptor();

    /**
     * update normal vector and depth
     */
    void UpdateNormalAndDepth();

    float GetMinDistanceInvariance();
    float GetMaxDistanceInvariance();

    int PredictScale(const float &currentDist, KeyFrame*pKF);
    int PredictScale(const float &currentDist, Frame* pF);

public:
    long unsigned int mnId;
    static long unsigned int nNextId;
    long int mnFirstKFid;
    long int mnFirstFrame;
    int nObs; // number of observations of mappoints, stereo+2/monocular+1

    // Variables used by the tracking
    float mTrackProjX;
    float mTrackProjY;
    float mTrackProjXR;
    bool mbTrackInView;
    int mnTrackScaleLevel;
    float mTrackViewCos;
    long unsigned int mnTrackReferenceForFrame;
    long unsigned int mnLastFrameSeen;

    // Variables used by local mapping
    long unsigned int mnBALocalForKF;
    long unsigned int mnFuseCandidateForKF;

    // Variables used by loop closing
    long unsigned int mnLoopPointForKF;
    long unsigned int mnCorrectedByKF;
    long unsigned int mnCorrectedReference;    
    cv::Mat mPosGBA;
    long unsigned int mnBAGlobalForKF;


    static std::mutex mGlobalMutex;

protected:    

     // Position in absolute coordinates
     cv::Mat mWorldPos;

     // Keyframes observing the point and associated index in keyframe
     std::map<KeyFrame*,size_t> mObservations;

     // Mean viewing direction
     cv::Mat mNormalVector;

     // Best descriptor to fast matching
     cv::Mat mDescriptor;

     // Reference KeyFrame
     KeyFrame* mpRefKF;

     // Tracking counters
     int mnVisible;
     int mnFound;

     // Bad flag (we do not currently erase MapPoint from memory)
     bool mbBad;
     MapPoint* mpReplaced;

     // Scale invariance distances
     float mfMinDistance;
     float mfMaxDistance;

     Map* mpMap;

     std::mutex mMutexPos;
     std::mutex mMutexFeatures;
};

} //namespace ORB_SLAM

#endif // MAPPOINT_H
