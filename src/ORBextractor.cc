/**
* This file is part of ORB-SLAM2.
* This file is based on the file orb.cpp from the OpenCV library (see BSD license below).
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
/**
* Software License Agreement (BSD License)
*
*  Copyright (c) 2009, Willow Garage, Inc.
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*
*/


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

#include "ORBextractor.h"


using namespace cv;
using namespace std;

namespace ORB_SLAM2
{

const int PATCH_SIZE = 31;
const int HALF_PATCH_SIZE = 15;
const int EDGE_THRESHOLD = 19;


/**
 * @brief
 * 灰度质心法（IC）计算特征的旋转
 * 
 * @return  计算的角度
 */
static float IC_Angle(const Mat& image, Point2f pt,  const vector<int> & u_max)
{
    int m_01 = 0, m_10 = 0;                                                 // 根据公式初始化图像块的矩

    const uchar* center = &image.at<uchar> (cvRound(pt.y), cvRound(pt.x));  // cvRound 返回跟参数最接近的整数值

    // Treat the center line differently, v=0   横坐标：-15-----+15
    // 我们要在一个圆域中算出m10和m01，计算步骤是先算出中间红线的m10，然后在平行于x轴算出m10和m01，一次计算相当于图像中的同个颜色的两个line。
    for (int u = -HALF_PATCH_SIZE; u <= HALF_PATCH_SIZE; ++u)
        m_10 += u * center[u];

    // Go line by line in the circuI853lar patch
    int step = (int)image.step1();                                          // opencv中概念，计算每行的元素个数
    for (int v = 1; v <= HALF_PATCH_SIZE; ++v)
    {
        // Proceed over the two lines
        int v_sum = 0;
        int d = u_max[v];
        for (int u = -d; u <= d; ++u)
        {
            int val_plus = center[u + v*step], val_minus = center[u - v*step];
            v_sum += (val_plus - val_minus);
            m_10 += u * (val_plus + val_minus);
        }
        m_01 += v * v_sum;
    }

    return fastAtan2((float)m_01, (float)m_10);                             // 返回计算的角度
}


const float factorPI = (float)(CV_PI/180.f);            // 弧度制和角度制之间的转换


static void computeOrbDescriptor(const KeyPoint& kpt,
                                 const Mat& img, const Point* pattern,
                                 uchar* desc)
{
    float angle = (float)kpt.angle*factorPI;            // arc
    float a = (float)cos(angle), b = (float)sin(angle); // 考虑关键点的方向

    const uchar* center = &img.at<uchar>(cvRound(kpt.pt.y), cvRound(kpt.pt.x));
    const int step = (int)img.step;

    // 注意这里在pattern里找点的时候，是利用关键点的方向信息进行了**旋转矫正**的
    #define GET_VALUE(idx) \
        center[cvRound(pattern[idx].x*b + pattern[idx].y*a)*step + \
               cvRound(pattern[idx].x*a - pattern[idx].y*b)]


    for (int i = 0; i < 32; ++i, pattern += 16)
    {
        int t0, t1, val;
        t0 = GET_VALUE(0); t1 = GET_VALUE(1);
        val = t0 < t1;
        t0 = GET_VALUE(2); t1 = GET_VALUE(3);
        val |= (t0 < t1) << 1;
        t0 = GET_VALUE(4); t1 = GET_VALUE(5);
        val |= (t0 < t1) << 2;
        t0 = GET_VALUE(6); t1 = GET_VALUE(7);
        val |= (t0 < t1) << 3;
        t0 = GET_VALUE(8); t1 = GET_VALUE(9);
        val |= (t0 < t1) << 4;
        t0 = GET_VALUE(10); t1 = GET_VALUE(11);
        val |= (t0 < t1) << 5;
        t0 = GET_VALUE(12); t1 = GET_VALUE(13);
        val |= (t0 < t1) << 6;
        t0 = GET_VALUE(14); t1 = GET_VALUE(15);
        val |= (t0 < t1) << 7;

        desc[i] = (uchar)val;
    }

    #undef GET_VALUE
}


static int bit_pattern_31_[256*4] =
{
    8,-3, 9,5/*mean (0), correlation (0)*/,
    4,2, 7,-12/*mean (1.12461e-05), correlation (0.0437584)*/,
    -11,9, -8,2/*mean (3.37382e-05), correlation (0.0617409)*/,
    7,-12, 12,-13/*mean (5.62303e-05), correlation (0.0636977)*/,
    2,-13, 2,12/*mean (0.000134953), correlation (0.085099)*/,
    1,-7, 1,6/*mean (0.000528565), correlation (0.0857175)*/,
    -2,-10, -2,-4/*mean (0.0188821), correlation (0.0985774)*/,
    -13,-13, -11,-8/*mean (0.0363135), correlation (0.0899616)*/,
    -13,-3, -12,-9/*mean (0.121806), correlation (0.099849)*/,
    10,4, 11,9/*mean (0.122065), correlation (0.093285)*/,
    -13,-8, -8,-9/*mean (0.162787), correlation (0.0942748)*/,
    -11,7, -9,12/*mean (0.21561), correlation (0.0974438)*/,
    7,7, 12,6/*mean (0.160583), correlation (0.130064)*/,
    -4,-5, -3,0/*mean (0.228171), correlation (0.132998)*/,
    -13,2, -12,-3/*mean (0.00997526), correlation (0.145926)*/,
    -9,0, -7,5/*mean (0.198234), correlation (0.143636)*/,
    12,-6, 12,-1/*mean (0.0676226), correlation (0.16689)*/,
    -3,6, -2,12/*mean (0.166847), correlation (0.171682)*/,
    -6,-13, -4,-8/*mean (0.101215), correlation (0.179716)*/,
    11,-13, 12,-8/*mean (0.200641), correlation (0.192279)*/,
    4,7, 5,1/*mean (0.205106), correlation (0.186848)*/,
    5,-3, 10,-3/*mean (0.234908), correlation (0.192319)*/,
    3,-7, 6,12/*mean (0.0709964), correlation (0.210872)*/,
    -8,-7, -6,-2/*mean (0.0939834), correlation (0.212589)*/,
    -2,11, -1,-10/*mean (0.127778), correlation (0.20866)*/,
    -13,12, -8,10/*mean (0.14783), correlation (0.206356)*/,
    -7,3, -5,-3/*mean (0.182141), correlation (0.198942)*/,
    -4,2, -3,7/*mean (0.188237), correlation (0.21384)*/,
    -10,-12, -6,11/*mean (0.14865), correlation (0.23571)*/,
    5,-12, 6,-7/*mean (0.222312), correlation (0.23324)*/,
    5,-6, 7,-1/*mean (0.229082), correlation (0.23389)*/,
    1,0, 4,-5/*mean (0.241577), correlation (0.215286)*/,
    9,11, 11,-13/*mean (0.00338507), correlation (0.251373)*/,
    4,7, 4,12/*mean (0.131005), correlation (0.257622)*/,
    2,-1, 4,4/*mean (0.152755), correlation (0.255205)*/,
    -4,-12, -2,7/*mean (0.182771), correlation (0.244867)*/,
    -8,-5, -7,-10/*mean (0.186898), correlation (0.23901)*/,
    4,11, 9,12/*mean (0.226226), correlation (0.258255)*/,
    0,-8, 1,-13/*mean (0.0897886), correlation (0.274827)*/,
    -13,-2, -8,2/*mean (0.148774), correlation (0.28065)*/,
    -3,-2, -2,3/*mean (0.153048), correlation (0.283063)*/,
    -6,9, -4,-9/*mean (0.169523), correlation (0.278248)*/,
    8,12, 10,7/*mean (0.225337), correlation (0.282851)*/,
    0,9, 1,3/*mean (0.226687), correlation (0.278734)*/,
    7,-5, 11,-10/*mean (0.00693882), correlation (0.305161)*/,
    -13,-6, -11,0/*mean (0.0227283), correlation (0.300181)*/,
    10,7, 12,1/*mean (0.125517), correlation (0.31089)*/,
    -6,-3, -6,12/*mean (0.131748), correlation (0.312779)*/,
    10,-9, 12,-4/*mean (0.144827), correlation (0.292797)*/,
    -13,8, -8,-12/*mean (0.149202), correlation (0.308918)*/,
    -13,0, -8,-4/*mean (0.160909), correlation (0.310013)*/,
    3,3, 7,8/*mean (0.177755), correlation (0.309394)*/,
    5,7, 10,-7/*mean (0.212337), correlation (0.310315)*/,
    -1,7, 1,-12/*mean (0.214429), correlation (0.311933)*/,
    3,-10, 5,6/*mean (0.235807), correlation (0.313104)*/,
    2,-4, 3,-10/*mean (0.00494827), correlation (0.344948)*/,
    -13,0, -13,5/*mean (0.0549145), correlation (0.344675)*/,
    -13,-7, -12,12/*mean (0.103385), correlation (0.342715)*/,
    -13,3, -11,8/*mean (0.134222), correlation (0.322922)*/,
    -7,12, -4,7/*mean (0.153284), correlation (0.337061)*/,
    6,-10, 12,8/*mean (0.154881), correlation (0.329257)*/,
    -9,-1, -7,-6/*mean (0.200967), correlation (0.33312)*/,
    -2,-5, 0,12/*mean (0.201518), correlation (0.340635)*/,
    -12,5, -7,5/*mean (0.207805), correlation (0.335631)*/,
    3,-10, 8,-13/*mean (0.224438), correlation (0.34504)*/,
    -7,-7, -4,5/*mean (0.239361), correlation (0.338053)*/,
    -3,-2, -1,-7/*mean (0.240744), correlation (0.344322)*/,
    2,9, 5,-11/*mean (0.242949), correlation (0.34145)*/,
    -11,-13, -5,-13/*mean (0.244028), correlation (0.336861)*/,
    -1,6, 0,-1/*mean (0.247571), correlation (0.343684)*/,
    5,-3, 5,2/*mean (0.000697256), correlation (0.357265)*/,
    -4,-13, -4,12/*mean (0.00213675), correlation (0.373827)*/,
    -9,-6, -9,6/*mean (0.0126856), correlation (0.373938)*/,
    -12,-10, -8,-4/*mean (0.0152497), correlation (0.364237)*/,
    10,2, 12,-3/*mean (0.0299933), correlation (0.345292)*/,
    7,12, 12,12/*mean (0.0307242), correlation (0.366299)*/,
    -7,-13, -6,5/*mean (0.0534975), correlation (0.368357)*/,
    -4,9, -3,4/*mean (0.099865), correlation (0.372276)*/,
    7,-1, 12,2/*mean (0.117083), correlation (0.364529)*/,
    -7,6, -5,1/*mean (0.126125), correlation (0.369606)*/,
    -13,11, -12,5/*mean (0.130364), correlation (0.358502)*/,
    -3,7, -2,-6/*mean (0.131691), correlation (0.375531)*/,
    7,-8, 12,-7/*mean (0.160166), correlation (0.379508)*/,
    -13,-7, -11,-12/*mean (0.167848), correlation (0.353343)*/,
    1,-3, 12,12/*mean (0.183378), correlation (0.371916)*/,
    2,-6, 3,0/*mean (0.228711), correlation (0.371761)*/,
    -4,3, -2,-13/*mean (0.247211), correlation (0.364063)*/,
    -1,-13, 1,9/*mean (0.249325), correlation (0.378139)*/,
    7,1, 8,-6/*mean (0.000652272), correlation (0.411682)*/,
    1,-1, 3,12/*mean (0.00248538), correlation (0.392988)*/,
    9,1, 12,6/*mean (0.0206815), correlation (0.386106)*/,
    -1,-9, -1,3/*mean (0.0364485), correlation (0.410752)*/,
    -13,-13, -10,5/*mean (0.0376068), correlation (0.398374)*/,
    7,7, 10,12/*mean (0.0424202), correlation (0.405663)*/,
    12,-5, 12,9/*mean (0.0942645), correlation (0.410422)*/,
    6,3, 7,11/*mean (0.1074), correlation (0.413224)*/,
    5,-13, 6,10/*mean (0.109256), correlation (0.408646)*/,
    2,-12, 2,3/*mean (0.131691), correlation (0.416076)*/,
    3,8, 4,-6/*mean (0.165081), correlation (0.417569)*/,
    2,6, 12,-13/*mean (0.171874), correlation (0.408471)*/,
    9,-12, 10,3/*mean (0.175146), correlation (0.41296)*/,
    -8,4, -7,9/*mean (0.183682), correlation (0.402956)*/,
    -11,12, -4,-6/*mean (0.184672), correlation (0.416125)*/,
    1,12, 2,-8/*mean (0.191487), correlation (0.386696)*/,
    6,-9, 7,-4/*mean (0.192668), correlation (0.394771)*/,
    2,3, 3,-2/*mean (0.200157), correlation (0.408303)*/,
    6,3, 11,0/*mean (0.204588), correlation (0.411762)*/,
    3,-3, 8,-8/*mean (0.205904), correlation (0.416294)*/,
    7,8, 9,3/*mean (0.213237), correlation (0.409306)*/,
    -11,-5, -6,-4/*mean (0.243444), correlation (0.395069)*/,
    -10,11, -5,10/*mean (0.247672), correlation (0.413392)*/,
    -5,-8, -3,12/*mean (0.24774), correlation (0.411416)*/,
    -10,5, -9,0/*mean (0.00213675), correlation (0.454003)*/,
    8,-1, 12,-6/*mean (0.0293635), correlation (0.455368)*/,
    4,-6, 6,-11/*mean (0.0404971), correlation (0.457393)*/,
    -10,12, -8,7/*mean (0.0481107), correlation (0.448364)*/,
    4,-2, 6,7/*mean (0.050641), correlation (0.455019)*/,
    -2,0, -2,12/*mean (0.0525978), correlation (0.44338)*/,
    -5,-8, -5,2/*mean (0.0629667), correlation (0.457096)*/,
    7,-6, 10,12/*mean (0.0653846), correlation (0.445623)*/,
    -9,-13, -8,-8/*mean (0.0858749), correlation (0.449789)*/,
    -5,-13, -5,-2/*mean (0.122402), correlation (0.450201)*/,
    8,-8, 9,-13/*mean (0.125416), correlation (0.453224)*/,
    -9,-11, -9,0/*mean (0.130128), correlation (0.458724)*/,
    1,-8, 1,-2/*mean (0.132467), correlation (0.440133)*/,
    7,-4, 9,1/*mean (0.132692), correlation (0.454)*/,
    -2,1, -1,-4/*mean (0.135695), correlation (0.455739)*/,
    11,-6, 12,-11/*mean (0.142904), correlation (0.446114)*/,
    -12,-9, -6,4/*mean (0.146165), correlation (0.451473)*/,
    3,7, 7,12/*mean (0.147627), correlation (0.456643)*/,
    5,5, 10,8/*mean (0.152901), correlation (0.455036)*/,
    0,-4, 2,8/*mean (0.167083), correlation (0.459315)*/,
    -9,12, -5,-13/*mean (0.173234), correlation (0.454706)*/,
    0,7, 2,12/*mean (0.18312), correlation (0.433855)*/,
    -1,2, 1,7/*mean (0.185504), correlation (0.443838)*/,
    5,11, 7,-9/*mean (0.185706), correlation (0.451123)*/,
    3,5, 6,-8/*mean (0.188968), correlation (0.455808)*/,
    -13,-4, -8,9/*mean (0.191667), correlation (0.459128)*/,
    -5,9, -3,-3/*mean (0.193196), correlation (0.458364)*/,
    -4,-7, -3,-12/*mean (0.196536), correlation (0.455782)*/,
    6,5, 8,0/*mean (0.1972), correlation (0.450481)*/,
    -7,6, -6,12/*mean (0.199438), correlation (0.458156)*/,
    -13,6, -5,-2/*mean (0.211224), correlation (0.449548)*/,
    1,-10, 3,10/*mean (0.211718), correlation (0.440606)*/,
    4,1, 8,-4/*mean (0.213034), correlation (0.443177)*/,
    -2,-2, 2,-13/*mean (0.234334), correlation (0.455304)*/,
    2,-12, 12,12/*mean (0.235684), correlation (0.443436)*/,
    -2,-13, 0,-6/*mean (0.237674), correlation (0.452525)*/,
    4,1, 9,3/*mean (0.23962), correlation (0.444824)*/,
    -6,-10, -3,-5/*mean (0.248459), correlation (0.439621)*/,
    -3,-13, -1,1/*mean (0.249505), correlation (0.456666)*/,
    7,5, 12,-11/*mean (0.00119208), correlation (0.495466)*/,
    4,-2, 5,-7/*mean (0.00372245), correlation (0.484214)*/,
    -13,9, -9,-5/*mean (0.00741116), correlation (0.499854)*/,
    7,1, 8,6/*mean (0.0208952), correlation (0.499773)*/,
    7,-8, 7,6/*mean (0.0220085), correlation (0.501609)*/,
    -7,-4, -7,1/*mean (0.0233806), correlation (0.496568)*/,
    -8,11, -7,-8/*mean (0.0236505), correlation (0.489719)*/,
    -13,6, -12,-8/*mean (0.0268781), correlation (0.503487)*/,
    2,4, 3,9/*mean (0.0323324), correlation (0.501938)*/,
    10,-5, 12,3/*mean (0.0399235), correlation (0.494029)*/,
    -6,-5, -6,7/*mean (0.0420153), correlation (0.486579)*/,
    8,-3, 9,-8/*mean (0.0548021), correlation (0.484237)*/,
    2,-12, 2,8/*mean (0.0616622), correlation (0.496642)*/,
    -11,-2, -10,3/*mean (0.0627755), correlation (0.498563)*/,
    -12,-13, -7,-9/*mean (0.0829622), correlation (0.495491)*/,
    -11,0, -10,-5/*mean (0.0843342), correlation (0.487146)*/,
    5,-3, 11,8/*mean (0.0929937), correlation (0.502315)*/,
    -2,-13, -1,12/*mean (0.113327), correlation (0.48941)*/,
    -1,-8, 0,9/*mean (0.132119), correlation (0.467268)*/,
    -13,-11, -12,-5/*mean (0.136269), correlation (0.498771)*/,
    -10,-2, -10,11/*mean (0.142173), correlation (0.498714)*/,
    -3,9, -2,-13/*mean (0.144141), correlation (0.491973)*/,
    2,-3, 3,2/*mean (0.14892), correlation (0.500782)*/,
    -9,-13, -4,0/*mean (0.150371), correlation (0.498211)*/,
    -4,6, -3,-10/*mean (0.152159), correlation (0.495547)*/,
    -4,12, -2,-7/*mean (0.156152), correlation (0.496925)*/,
    -6,-11, -4,9/*mean (0.15749), correlation (0.499222)*/,
    6,-3, 6,11/*mean (0.159211), correlation (0.503821)*/,
    -13,11, -5,5/*mean (0.162427), correlation (0.501907)*/,
    11,11, 12,6/*mean (0.16652), correlation (0.497632)*/,
    7,-5, 12,-2/*mean (0.169141), correlation (0.484474)*/,
    -1,12, 0,7/*mean (0.169456), correlation (0.495339)*/,
    -4,-8, -3,-2/*mean (0.171457), correlation (0.487251)*/,
    -7,1, -6,7/*mean (0.175), correlation (0.500024)*/,
    -13,-12, -8,-13/*mean (0.175866), correlation (0.497523)*/,
    -7,-2, -6,-8/*mean (0.178273), correlation (0.501854)*/,
    -8,5, -6,-9/*mean (0.181107), correlation (0.494888)*/,
    -5,-1, -4,5/*mean (0.190227), correlation (0.482557)*/,
    -13,7, -8,10/*mean (0.196739), correlation (0.496503)*/,
    1,5, 5,-13/*mean (0.19973), correlation (0.499759)*/,
    1,0, 10,-13/*mean (0.204465), correlation (0.49873)*/,
    9,12, 10,-1/*mean (0.209334), correlation (0.49063)*/,
    5,-8, 10,-9/*mean (0.211134), correlation (0.503011)*/,
    -1,11, 1,-13/*mean (0.212), correlation (0.499414)*/,
    -9,-3, -6,2/*mean (0.212168), correlation (0.480739)*/,
    -1,-10, 1,12/*mean (0.212731), correlation (0.502523)*/,
    -13,1, -8,-10/*mean (0.21327), correlation (0.489786)*/,
    8,-11, 10,-6/*mean (0.214159), correlation (0.488246)*/,
    2,-13, 3,-6/*mean (0.216993), correlation (0.50287)*/,
    7,-13, 12,-9/*mean (0.223639), correlation (0.470502)*/,
    -10,-10, -5,-7/*mean (0.224089), correlation (0.500852)*/,
    -10,-8, -8,-13/*mean (0.228666), correlation (0.502629)*/,
    4,-6, 8,5/*mean (0.22906), correlation (0.498305)*/,
    3,12, 8,-13/*mean (0.233378), correlation (0.503825)*/,
    -4,2, -3,-3/*mean (0.234323), correlation (0.476692)*/,
    5,-13, 10,-12/*mean (0.236392), correlation (0.475462)*/,
    4,-13, 5,-1/*mean (0.236842), correlation (0.504132)*/,
    -9,9, -4,3/*mean (0.236977), correlation (0.497739)*/,
    0,3, 3,-9/*mean (0.24314), correlation (0.499398)*/,
    -12,1, -6,1/*mean (0.243297), correlation (0.489447)*/,
    3,2, 4,-8/*mean (0.00155196), correlation (0.553496)*/,
    -10,-10, -10,9/*mean (0.00239541), correlation (0.54297)*/,
    8,-13, 12,12/*mean (0.0034413), correlation (0.544361)*/,
    -8,-12, -6,-5/*mean (0.003565), correlation (0.551225)*/,
    2,2, 3,7/*mean (0.00835583), correlation (0.55285)*/,
    10,6, 11,-8/*mean (0.00885065), correlation (0.540913)*/,
    6,8, 8,-12/*mean (0.0101552), correlation (0.551085)*/,
    -7,10, -6,5/*mean (0.0102227), correlation (0.533635)*/,
    -3,-9, -3,9/*mean (0.0110211), correlation (0.543121)*/,
    -1,-13, -1,5/*mean (0.0113473), correlation (0.550173)*/,
    -3,-7, -3,4/*mean (0.0140913), correlation (0.554774)*/,
    -8,-2, -8,3/*mean (0.017049), correlation (0.55461)*/,
    4,2, 12,12/*mean (0.01778), correlation (0.546921)*/,
    2,-5, 3,11/*mean (0.0224022), correlation (0.549667)*/,
    6,-9, 11,-13/*mean (0.029161), correlation (0.546295)*/,
    3,-1, 7,12/*mean (0.0303081), correlation (0.548599)*/,
    11,-1, 12,4/*mean (0.0355151), correlation (0.523943)*/,
    -3,0, -3,6/*mean (0.0417904), correlation (0.543395)*/,
    4,-11, 4,12/*mean (0.0487292), correlation (0.542818)*/,
    2,-4, 2,1/*mean (0.0575124), correlation (0.554888)*/,
    -10,-6, -8,1/*mean (0.0594242), correlation (0.544026)*/,
    -13,7, -11,1/*mean (0.0597391), correlation (0.550524)*/,
    -13,12, -11,-13/*mean (0.0608974), correlation (0.55383)*/,
    6,0, 11,-13/*mean (0.065126), correlation (0.552006)*/,
    0,-1, 1,4/*mean (0.074224), correlation (0.546372)*/,
    -13,3, -9,-2/*mean (0.0808592), correlation (0.554875)*/,
    -9,8, -6,-3/*mean (0.0883378), correlation (0.551178)*/,
    -13,-6, -8,-2/*mean (0.0901035), correlation (0.548446)*/,
    5,-9, 8,10/*mean (0.0949843), correlation (0.554694)*/,
    2,7, 3,-9/*mean (0.0994152), correlation (0.550979)*/,
    -1,-6, -1,-1/*mean (0.10045), correlation (0.552714)*/,
    9,5, 11,-2/*mean (0.100686), correlation (0.552594)*/,
    11,-3, 12,-8/*mean (0.101091), correlation (0.532394)*/,
    3,0, 3,5/*mean (0.101147), correlation (0.525576)*/,
    -1,4, 0,10/*mean (0.105263), correlation (0.531498)*/,
    3,-6, 4,5/*mean (0.110785), correlation (0.540491)*/,
    -13,0, -10,5/*mean (0.112798), correlation (0.536582)*/,
    5,8, 12,11/*mean (0.114181), correlation (0.555793)*/,
    8,9, 9,-6/*mean (0.117431), correlation (0.553763)*/,
    7,-4, 8,-12/*mean (0.118522), correlation (0.553452)*/,
    -10,4, -10,9/*mean (0.12094), correlation (0.554785)*/,
    7,3, 12,4/*mean (0.122582), correlation (0.555825)*/,
    9,-7, 10,-2/*mean (0.124978), correlation (0.549846)*/,
    7,0, 12,-2/*mean (0.127002), correlation (0.537452)*/,
    -1,-6, 0,-11/*mean (0.127148), correlation (0.547401)*/
};


/**
 * @brief 
 * 提取特征前的准备工作
 * 
 * @param nfeatures     ORB特征点数量
 * @param scaleFactor   相邻层的放大倍数
 * @param nlevels       层数
 * @param iniThFAST     提取FAST角点时初始阈值
 * @param minThFAST     提取FAST角点时,更小的阈值
 * @note 设置两个阈值的原因是在FAST提取角点进行分块后有可能在某个块中在原始阈值情况下提取不到角点，使用更小的阈值进一步提取
 */
ORBextractor::ORBextractor(int _nfeatures, float _scaleFactor, int _nlevels,
         int _iniThFAST, int _minThFAST):
    nfeatures(_nfeatures), scaleFactor(_scaleFactor), nlevels(_nlevels),
    iniThFAST(_iniThFAST), minThFAST(_minThFAST)
{
    mvScaleFactor.resize(nlevels);
    mvLevelSigma2.resize(nlevels);
    mvScaleFactor[0]=1.0f;              // 初始化mvScaleFactor
    mvLevelSigma2[0]=1.0f;              // 初始化mvLevelSigma2

    // 根据scaleFactor计算每层的mvScaleFactor，其值单调递增
    for(int i=1; i<nlevels; i++)
    {                                                       // 0    1    2     3      4
        mvScaleFactor[i]=mvScaleFactor[i-1]*scaleFactor;    // 1.0, 0.5, 0.25, 0.125, 0.0625, ...
        mvLevelSigma2[i]=mvScaleFactor[i]*mvScaleFactor[i];
    }

    mvInvScaleFactor.resize(nlevels);
    mvInvLevelSigma2.resize(nlevels);

    // 计算mvInvScaleFactor，mvInvLevelSigma2
    for(int i=0; i<nlevels; i++)
    {
        mvInvScaleFactor[i]=1.0f/mvScaleFactor[i];
        mvInvLevelSigma2[i]=1.0f/mvLevelSigma2[i];
    }

    mvImagePyramid.resize(nlevels);     // 图像金字塔, 存放各层的图片

    /**
     * 对于缩放的每层高斯金字塔图像，计算其对应每层待提取特征的**数量**, 放入 mnFeaturesPerLevel 中,
     * 使得每层特征点的数列成等比数列递减
     */
    mnFeaturesPerLevel.resize(nlevels);
    float factor = 1.0f / scaleFactor;
    float nDesiredFeaturesPerScale = nfeatures*(1 - factor)/(1 - (float)pow((double)factor, (double)nlevels));

    int sumFeatures = 0;
    for( int level = 0; level < nlevels-1; level++ )
    {
        mnFeaturesPerLevel[level] = cvRound(nDesiredFeaturesPerScale);      // number of features per level
        sumFeatures += mnFeaturesPerLevel[level];                           // accumulate the number of features
        nDesiredFeaturesPerScale *= factor;
    }
    mnFeaturesPerLevel[nlevels-1] = std::max(nfeatures - sumFeatures, 0);   // the lowest level, smallest size

    // 准备计算关键点keypoint的brief描述子时所需要的pattern, 这个pattern一共有512个点对
    const int npoints = 512;

    /** 
     * bit_pattern_31_是一个1024维的数组，其信息是512对点对的相对中心点的像素坐标
     * 这里将bit_pattern_31_里的信息以Point的形式存储在了std::vector<cv::Point> pattern里
     * 最后pattern储存了512个Point
     */
    const Point* pattern0 = (const Point*)bit_pattern_31_;
    std::copy(pattern0, pattern0 + npoints, std::back_inserter(pattern));

    // This is for orientation
    // pre-compute the end of a row in a circular patch
    /**
     * 这里是为了计算关键点方向的准备工作
     * 我们是要在以关键点keypoint像素坐标点为中心的直径为PATCH_SIZE，半径为HALF_PATCH_SIZE的patch圆内计算关键点keypoint的方向
     * 那如何描述这个patch圆的范围呢？这里选择的是储存不同v所对应的的umax来描述这个patch圆的范围
     * 
     * 为什么要计算umax？
     * 我们知道，ORB特征点是由FAST关键点和BRIEF描述子组成的
     * ORB特征点的关键点告诉我们ORB特征点的位置信息。在后面我们还需要计算ORB特征点的方向
     * 而ORB特征点是根据像素坐标点为中心的直径为PATCH_SIZE，半径为HALF_PATCH_SIZE的patch圆内计算出来的
     * 
     * 当0<v<vmax时，我们通过三角形ABO，利用勾股定理计算umax[v]
     * 当v>vmax，我们就利用圆的对称性，计算余下的umax[v]
     */
    umax.resize(HALF_PATCH_SIZE + 1);

    int v, v0, vmax = cvFloor(HALF_PATCH_SIZE * sqrt(2.f) / 2 + 1);
    int vmin = cvCeil(HALF_PATCH_SIZE * sqrt(2.f) / 2);
    const double hp2 = HALF_PATCH_SIZE*HALF_PATCH_SIZE;
    for (v = 0; v <= vmax; ++v)
        umax[v] = cvRound(sqrt(hp2 - v * v));

    // Make sure we are symmetric
    // 其实是算当v=vmax至HALF_PATCH_SIZE时的umax[v]
    for (v = HALF_PATCH_SIZE, v0 = 0; v >= vmin; --v)
    {
        while (umax[v0] == umax[v0 + 1])
            ++v0;
        umax[v] = v0;
        ++v0;
    }
}

static void computeOrientation(const Mat& image, vector<KeyPoint>& keypoints, const vector<int>& umax)
{
    for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
         keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint)
    {
        keypoint->angle = IC_Angle(image, keypoint->pt, umax);
    }
}


/**
 * 将ExtractorNode内的关键点按照其位置分配给n1~n4
 * divide once
 */
void ExtractorNode::DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4)
{
    const int halfX = ceil(static_cast<float>(UR.x-UL.x)/2);
    const int halfY = ceil(static_cast<float>(BR.y-UL.y)/2);

    /**
     *       UL                  UR
     *          _________________
     *         |        |        |
     *         |   n1   |   n2   |
     *         |________|________|
     *         |        |        |
     *         |   n3   |   n4   |
     *         |________|________|
     * 
     *       BL                  BR
     */
    // Define boundaries of childs
    n1.UL = UL;
    n1.UR = cv::Point2i(UL.x+halfX,UL.y);
    n1.BL = cv::Point2i(UL.x,UL.y+halfY);
    n1.BR = cv::Point2i(UL.x+halfX,UL.y+halfY);
    n1.vKeys.reserve(vKeys.size());

    n2.UL = n1.UR;
    n2.UR = UR;
    n2.BL = n1.BR;
    n2.BR = cv::Point2i(UR.x,UL.y+halfY);
    n2.vKeys.reserve(vKeys.size());

    n3.UL = n1.BL;
    n3.UR = n1.BR;
    n3.BL = BL;
    n3.BR = cv::Point2i(n1.BR.x,BL.y);
    n3.vKeys.reserve(vKeys.size());

    n4.UL = n3.UR;
    n4.UR = n2.BR;
    n4.BL = n3.BR;
    n4.BR = BR;
    n4.vKeys.reserve(vKeys.size());

    // Associate points to childs
    for(size_t i=0;i<vKeys.size();i++)
    {
        const cv::KeyPoint &kp = vKeys[i];
        if(kp.pt.x<n1.UR.x)
        {
            if(kp.pt.y<n1.BR.y)
                n1.vKeys.push_back(kp);
            else
                n3.vKeys.push_back(kp);
        }
        else if(kp.pt.y<n1.BR.y)
            n2.vKeys.push_back(kp);
        else
            n4.vKeys.push_back(kp);
    }

    // check whether or not there is only one node in every subdivided area
    if(n1.vKeys.size()==1)
        n1.bNoMore = true;
    if(n2.vKeys.size()==1)
        n2.bNoMore = true;
    if(n3.vKeys.size()==1)
        n3.bNoMore = true;
    if(n4.vKeys.size()==1)
        n4.bNoMore = true;

    /**
     * n1.UL
     * n1.UR
     * n1.BL
     * n1.BR
     * n1.vKeys
     * n1.bNoMore
     */
}


/**
 * @brief
 * 利用四叉树来筛选提取的特征点，使得筛选后的特征点数量在预计的范围，并且分布均匀
 * 
 * 四叉树的数据模型
 * https://blog.csdn.net/weixin_38636815/article/details/81981305
 * 在ORB-SLAM中是定义了一个list用来存储四叉树中的结点 std::vector<ExtractNode> lNodes;
 * 如果一个结点满足**被分割的条件**那么将他新分割出来的四个结点依次从头部插入到list，
 * 每插入一个新的结点就让list的迭代器指向list中的第一个元素，四个新结点插入完毕之后将母结点删除
 * 
 * 然后是从list头部开始依次遍历所有结点，判断哪个节点要被继续分割，
 * 如果需要继续分割则按照上面的方法完成新结点的存储和母结点的删除，
 * 因为每插入一个结点都会让迭代器指向新插入的结点，所以每次都是从list的头部开始遍历，
 * 所以刚被分割出来的结点如果满足分割条件会被紧接着继续分割，而最早生成的结点要越晚被分割
 * 
 * 为什么要将四叉树应用到ORB-SLAM中，听别人说是为了管理图像中提取的特征点，那这个四叉树到底在特征点的管理中起到了什么作用呢？
 * 程序中，有一个环节是将创建好的结点按照**结点中包含的特征点的个数**从小到大来对结点进行排序**，
 * 然后是**优先**对那些包含特征点相对较少的结点**进行分割**，因为结点中包含的特征点少，分割的次数也就少了，
 * 并且倘若还没有遍历完所有的结点得到的特征点的个数就已经达到了每层图像提取特征点的要求，
 * 那么就不必对那些包含特征点多的结点进行分割了，一个是分割起来费劲，再者，这个结点中包含比同等层次的结点要多的特征点，
 * 说明这里面的特征点有“集会滋事”的嫌疑， 毕竟我们希望最终得到的特征点尽可能的均匀的分布在图像中。
 * 
 * 接下来，就要对从上面每个结点中挑选出response最大的作为这个结点的代表保留下来
 * 
 * 
 * 
 * https://blog.csdn.net/u014797790/article/details/79776339
 * 计算fast特征点并进行筛选
 * 先用(maxX-minX)/(maxY-minY)来确定四叉数有几个初始节点，这里有 bug，如果输入的是一张宽高比小于0.5 的图像，
 * nIni 计算得到 0，下一步计算 hX 会报错，例如round(640/480)=1,所以只有一个初始节点，(UL,UR,BL,BR)就会分布到被裁掉边界后的图像的四个角。
 * 把所有的关键点分配给属于它的节点，当节点所分配到的关键点的个数为1时就不再进行分裂，当节点没有分配到关键点时就删除此节点。
 * 再根据兴趣点分布,利用四叉树方法对图像进行划分区域，当bFinish的值为true时就不再进行区域划分，首先对目前的区域进行划分，
 * 
 * 把每次划分得到的有关键点的子区域设为新的节点，将 nToExpand 参数加一，并插入到节点列表的前边，删除掉其父节点。
 * **只要新节点中的关键点的个数超过一个，就继续划分**，继续插入列表前面，继续删除父节点，直到划分的子区域中的关键点的个数是一个，
 * 然后迭代器加以移动到下一个节点，继续划分区域。
 * 
 * 当划分的区域(即节点)的个数大于关键点的个数或者分裂过程没有增加节点的个数时就将bFinish的值设为true，不再进行划分。
 * 如果以上条件没有满足，但是满足((int)lNodes.size()+nToExpand*3)>N，则说明将所有节点再分裂一次可以达到要求。
 * vSizeAndPointerToNode 是前面分裂出来的子节点（n1, n2, n3, n4）中可以分裂的节点。
 * 按照它们特征点的排序，先从特征点多的开始分裂，分裂的结果继续存储在 lNodes 中。每分裂一个节点都会进行一次判断，
 * 如果 lNodes 中的节点数量大于所需要的特征点数量，退出整个 while(!bFinish) 循环，如果进行了一次分裂，
 * 并没有增加节点数量，不玩了，退出整个 while(!bFinish) 循环。取出每一个节点(每个区域)对应的最大响应点，即我们确定的特征点
 * 
 * 
 * 
 * https://blog.csdn.net/qq_25458977/article/details/103915715
 * 主要思路是将该区域作为均分为n大块，然后作为n个根节点，遍历每一个节点，
 * 如果该节点有多于一个的特征点keypoints，则将该节点所处的区域均匀划分成四份，依次类推，
 * 直到每个子节点区域只含有一个keypoints，如果该节点区域没有kp则会被erase掉
 */
vector<cv::KeyPoint> ORBextractor::DistributeOctTree(const vector<cv::KeyPoint>& vToDistributeKeys, const int &minX,
                                       const int &maxX, const int &minY, const int &maxY, const int &N, const int &level)
{
    // Compute how many initial nodes
    // 根节点的数量nIni是根据边界的宽高比值确定的
    // vpIniNode: vector[nIni,1]
    const int nIni = round(static_cast<float>(maxX-minX)/(maxY-minY));

    // node
    // 确定根节点的数量nIni后，再计算**根节点的领域范围**
    // 计算根节点的X范围，其Y轴范围就是(maxY-minY)
    // 水平划分格子的宽度
    const float hX = static_cast<float>(maxX-minX)/nIni;

    // list
    // 新建一个双向链表，其实后面大部分操作是改变这个
    // 不断分裂 lNodes 中的节点，直至 lNodes 节点数量大于等于需要的特征点数(N),或者 lNodes 中节点所有节点的关键点数量都为1
    list<ExtractorNode> lNodes;

    vector<ExtractorNode*> vpIniNodes;      // vpIniNodes 在将 vToDistributeKeys 分配到父节点中时起索引作用
    vpIniNodes.resize(nIni);

    // 遍历每个根节点，计算根节点领域的四个点UL，UR，BL，BR并其存入 lNodes 和 vpIniNodes
    /**
     *                      hX
     *            |<------------------->|
     *          UL _____________________ UR
     *            |          |          |
     *            |          |          |
     *            |          |          |
     *            |__________|__________|
     *            |                     |
     *            |       vpIniNodes[i] |
     *            |          |          |
     *            |__________|__________|
     *          BL                       BR
     */
    for(int i=0; i<nIni; i++)
    {
        ExtractorNode ni;
        ni.UL = cv::Point2i(hX*static_cast<float>(i),0);
        ni.UR = cv::Point2i(hX*static_cast<float>(i+1),0);
        ni.BL = cv::Point2i(ni.UL.x,maxY-minY);
        ni.BR = cv::Point2i(ni.UR.x,maxY-minY);
        ni.vKeys.reserve(vToDistributeKeys.size());

        lNodes.push_back(ni);
        vpIniNodes[i] = &lNodes.back();
    }

    // Associate points to childs
    // 将 vToDistributeKeys 按照位置分配在**根节点**中
    for(size_t i=0;i<vToDistributeKeys.size();i++)
    {
        const cv::KeyPoint &kp = vToDistributeKeys[i];  // KeyPoints
        vpIniNodes[kp.pt.x/hX]->vKeys.push_back(kp);    // 21/10 = 2
    }

    // 现在lNodes中只放有根节点
    list<ExtractorNode>::iterator lit = lNodes.begin();
    while(lit!=lNodes.end())
    {
        if(lit->vKeys.size()==1)
        {
            lit->bNoMore=true;          // 如果根节点中关键点个数为1, 这个节点不能再分裂了
            lit++;
        }
        else if(lit->vKeys.empty())
            lit = lNodes.erase(lit);    // 如果为空，那么就删除这个节点
        else
            lit++;
    }

    // 后面大循环结束标志位
    bool bFinish = false;

    int iteration = 0;

    vector<pair<int,ExtractorNode*> > vSizeAndPointerToNode;
    vSizeAndPointerToNode.reserve(lNodes.size()*4); // four children

    // 现在 lNodes 里只有根节点，且每个节点的关键点数量都不为0
    // 根据兴趣点分布,利用4叉树方法对图像进行**划分区域**
    while(!bFinish)
    {
        iteration++;

        int prevSize = lNodes.size();               // size of current nodes

        lit = lNodes.begin();

        int nToExpand = 0;                          // 待分裂的节点数 nToExpand

        vSizeAndPointerToNode.clear();

        // 遍历双向链表 lNodes 中的节点
        // 将目前的子区域经**行划分**
        while(lit!=lNodes.end())
        {
            if(lit->bNoMore)                        // bool, true
            {
                // If node only contains one point do not subdivide and continue
                // 如果此节点包含的关键点数量为一，那么就不分裂
                lit++;
                continue;
            }
            else
            {
                // If more than one point, subdivide
                // 如果此节点关键点数量大于1，那么就分裂此节
                // 将此节点中的关键点通过 DivideNode 分配到n1,n2,n3,n4, 并且将此被分裂的节点删除
                // https://blog.csdn.net/weixin_38636815/article/details/81981305
                /**
                 * ni: storge position
                 *  ni.UL
                 *  ni.UR
                 *  ni.BL
                 *  ni.BR
                 *  ni.vKeys    storge KeyPoints in the area
                 *  ni.bNoMore
                 * 
                 * vSizeAndPointerToNode: storge Size And Pointer of nodes
                 */
                ExtractorNode n1,n2,n3,n4;          // four subnodes
                lit->DivideNode(n1,n2,n3,n4);       // divide once

                // Add childs if they contain points
                // check each area
                // 如果节点n1的关键点数量大于0，将其插入 lNodes 的**前面**
                // 如果节点n1的关键点数量大于1，将其插入 vSizeAndPointerToNode 中表示**待分裂**
                // 后面n2,n3,n4以此类推
                if(n1.vKeys.size()>0)
                {
                    lNodes.push_front(n1);
                    if(n1.vKeys.size()>1)   // 待分裂的节点 vSizeAndPointerToNode, size of node > 1
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(),&lNodes.front()));    // Size And Pointer
                        lNodes.front().lit = lNodes.begin();                                            // update the iterator Pointer
                    }
                }
                if(n2.vKeys.size()>0)
                {
                    lNodes.push_front(n2);
                    if(n2.vKeys.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if(n3.vKeys.size()>0)
                {
                    lNodes.push_front(n3);
                    if(n3.vKeys.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if(n4.vKeys.size()>0)
                {
                    lNodes.push_front(n4);
                    if(n4.vKeys.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }

                lit=lNodes.erase(lit);      // earse parent node which has been replaced by child nodes
                continue;
            }
        }       

        // Finish if there are more nodes than required features
        // or all nodes contain **just one point**
        // lNodes 节点数量大于等于需要的特征点数(N),或者 lNodes 中节点所有节点的关键点数量都为1，
        // 标志 bFinish = true ，结束循环
        if((int)lNodes.size()>=N || (int)lNodes.size()==prevSize)
        {
            bFinish = true;
        }
        // 待分裂的节点数 vSizeAndPointerToNode 分裂后，一般会增加 nToExpand*3 ，
        // 如果增加 nToExpand*3 后大于N，则进入此分支
        // 当再划分之后所有的Node数大于要求数目时
        else if(((int)lNodes.size()+nToExpand*3)>N)     // +4-1 = 3
        {

            while(!bFinish)
            {

                prevSize = lNodes.size();

                vector<pair<int,ExtractorNode*> > vPrevSizeAndPointerToNode = vSizeAndPointerToNode;
                vSizeAndPointerToNode.clear();

                // 根据 vPrevSizeAndPointerToNode.first 大小来排序，也就是根据**待分裂节点**的关键点**数量大小**来有小到大排序
                // 对需要划分的部分进行排序, 即对兴趣点数较多的区域进行划分
                sort(vPrevSizeAndPointerToNode.begin(),vPrevSizeAndPointerToNode.end());

                // 倒叙遍历vPrevSizeAndPointerToNode，也先分裂待分裂节点的关键点数量大的
                for(int j=vPrevSizeAndPointerToNode.size()-1;j>=0;j--)
                {
                    ExtractorNode n1,n2,n3,n4;
                    vPrevSizeAndPointerToNode[j].second->DivideNode(n1,n2,n3,n4);

                    // Add childs if they contain points
                    if(n1.vKeys.size()>0)
                    {
                        lNodes.push_front(n1);
                        if(n1.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n2.vKeys.size()>0)
                    {
                        lNodes.push_front(n2);
                        if(n2.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n3.vKeys.size()>0)
                    {
                        lNodes.push_front(n3);
                        if(n3.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n4.vKeys.size()>0)
                    {
                        lNodes.push_front(n4);
                        if(n4.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }

                    lNodes.erase(vPrevSizeAndPointerToNode[j].second->lit);     // 删除被分裂的节点

                    if((int)lNodes.size()>=N)
                        break;
                }

                if((int)lNodes.size()>=N || (int)lNodes.size()==prevSize)
                    bFinish = true;

            }
        }
    }

    // Retain the best point in each node
    // 取出每个节点中响应最大的特征点
    vector<cv::KeyPoint> vResultKeys;
    vResultKeys.reserve(nfeatures);
    for(list<ExtractorNode>::iterator lit=lNodes.begin(); lit!=lNodes.end(); lit++)
    {
        vector<cv::KeyPoint> &vNodeKeys = lit->vKeys;
        cv::KeyPoint* pKP = &vNodeKeys[0];
        float maxResponse = pKP->response;

        for(size_t k=1;k<vNodeKeys.size();k++)
        {
            if(vNodeKeys[k].response>maxResponse)
            {
                pKP = &vNodeKeys[k];
                maxResponse = vNodeKeys[k].response;
            }
        }

        vResultKeys.push_back(*pKP);
    }

    return vResultKeys;
}


/**
 * @brief
 * 利用四叉树提取高斯金字塔中每层图像的orb关键点
 * 
 * 提取出来的关键点不是以四叉树的形式储存的
 * 四叉树只是被利用使得**提取的关键点均匀分布**，并且用到了非极大值抑制(Non-maximum suppression, NMS)算法
 * 
 * 对影像金字塔中的每一层图像进行特征点的计算。具体计算过程是将影像网格分割成小区域，每一个小区域独立使用FAST角点检测
 * 检测完成之后使用DistributeOcTree函数对检测到所有的角点进行筛选，使得角点分布均匀
 * 
 * @param allKeypoints  提取图像金字塔中各层图像的关键点并储存在allKeypoints里
 */
void ORBextractor::ComputeKeyPointsOctTree(vector<vector<KeyPoint> >& allKeypoints)
{
    allKeypoints.resize(nlevels);

    const float W = 30;     // 暂定的分割窗口的大小

    // 对高斯金字塔 mvImagePyramid 中每层图像提取orb特征点
    for (int level = 0; level < nlevels; ++level)
    {
        // 1.先是计算提取关键点的**边界**，太边缘的地方放弃提取关键点
        const int minBorderX = EDGE_THRESHOLD-3;
        const int minBorderY = minBorderX;
        const int maxBorderX = mvImagePyramid[level].cols-EDGE_THRESHOLD+3;
        const int maxBorderY = mvImagePyramid[level].rows-EDGE_THRESHOLD+3;

        // 用这个存储待筛选的orb
        vector<cv::KeyPoint> vToDistributeKeys;             
        vToDistributeKeys.reserve(nfeatures*10);

        // 计算边界宽度和高度
        const float width = (maxBorderX-minBorderX);
        const float height = (maxBorderY-minBorderY);

        // 将原始图片分割的行数和列数
        const int nCols = width/W;
        const int nRows = height/W;

        // 2.将**图像分割**为WxW的小图片，遍历这些分割出来的小图片并提取其关键点
        // 实际分割窗口的大小
        const int wCell = ceil(width/nCols);
        const int hCell = ceil(height/nRows);

        // 使用两个for循环遍历每个窗口，首先先计算窗口的四个坐标
		// 遍历每行
        for(int i=0; i<nRows; i++)
        {
            /**
             *    [iniX,iniY]        [maxX,iniY]
             *          __________________
             *         |                  |
             *         |                  |
             *         |                  |
             *         |                  |
             *         |__________________|
             * 
             *    [iniX,maxY]        [maxX,maxY]
             *           
             */
            // iniY,maxY为窗口的行 上坐标和下坐标
        	// 计算(iniY,maxY)坐标
            const float iniY =minBorderY+i*hCell;
            float maxY = iniY+hCell+6;              // 这里注意窗口之间有6行的重叠

            if(iniY>=maxBorderY-3)                  // 窗口的行上坐标超出边界，则放弃此行
                continue;
            if(maxY>maxBorderY)                     // 窗口的行下坐标超出边界，则将窗口的行下坐标设置为边界
                maxY = maxBorderY;

            // 遍历每列
            for(int j=0; j<nCols; j++)
            {   
                // iniX,maxX为窗口的列 左坐标和右坐标
            	// 计算(iniX,maxX)坐标
                const float iniX =minBorderX+j*wCell;
                float maxX = iniX+wCell+6;          // 这里注意窗口之间有6列的重叠

                /**
                 * j = 0
                 * iniX = minBorderX
                 * maxX = iniX + wCell + 6
                 * 
                 * j = 1
                 * iniX = minBorderX + wCell
                 * maxX = iniX + wCell + 6
                 * 
                 * j = 2
                 * iniX = minBorderX + 2*wCell
                 * maxX = iniX + wCell + 6
                 * 
                 * ...
                 * 
                 * crop image to patch:
                 * level层的图片中行范围(iniY,maxY),列范围(iniX,maxX)
                 */

                if(iniX>=maxBorderX-6)              // 窗口的列左坐标超出边界，则放弃此列
                    continue;
                if(maxX>maxBorderX)                 // 窗口的列右坐标超出边界，则将窗口的列右坐标设置为边界
                    maxX = maxBorderX;

                // 对每个小块进行FAST兴趣点能提取，并将提取到的特征点保存在这个cell对应的 vKeysCell 中
                vector<cv::KeyPoint> vKeysCell;     // 每个小窗里的关键点KeyPoint将存在这里

                /**
                 * KeyPoint的构造是首先在FAST中完成的
                 * 
                 * 提取FAST角点
                 * 输入参数
                 * mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX): level层的图片中行范围(iniY,maxY),列范围(iniX,maxX)的截图
                 * vKeysCell    储存提取的fast关键点
                 * iniThFAST    提取角点的阈值
                 * true         是否开启非极大值抑制算法
                 */
                FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX), vKeysCell,iniThFAST,true);

                if(vKeysCell.empty())
                {
                    // 如果没有找到FAST关键点，就降低阈值重新计算FAST
                    FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX), vKeysCell,minThFAST,true);
                }

                // 如果找到的点不为空，就加入到 vToDistributeKeys
                if(!vKeysCell.empty())
                {
                    for(vector<cv::KeyPoint>::iterator vit=vKeysCell.begin(); vit!=vKeysCell.end();vit++)
                    {
                        /**
                         * 根据前面的行列计算**实际的位置**
                         * in patch position
                         * [(*vit).pt.x, (*vit).pt.y]
                         */
                        (*vit).pt.x+=j*wCell;
                        (*vit).pt.y+=i*hCell;
                        vToDistributeKeys.push_back(*vit);
                    }
                }

            }
        }

        // 3.将提取的关键点交给 DistributeOctTree() 利用四叉树以及非极大值抑制算法进行**筛选**
        vector<KeyPoint> & keypoints = allKeypoints[level]; // 经 DistributeOctTree 筛选后的关键点存储在这里
        keypoints.reserve(nfeatures);

        // 筛选 vToDistributeKeys 中的关键点
        keypoints = DistributeOctTree(vToDistributeKeys, minBorderX, maxBorderX,
                                      minBorderY, maxBorderY,mnFeaturesPerLevel[level], level);

        
        // 4.计算关键点的相关信息：像素位置，patch尺度，方向，所在高斯金字塔层数
        /**
         * 计算在本层提取出的关键点对应的Patch大小，称为scaledPatchSize
         * 你想想，本层的图像是缩小的，而你在本层提取的orb特征点，计算orb的方向，描述子的时候根据
         * 的PATCH大小依旧是PATCH_SIZE
         * 而你在本层提取的orb是要恢复到原始图像上去的，所以其特征点的size（代表特征点的尺度信息）需要放大
         */
        const int scaledPatchSize = PATCH_SIZE*mvScaleFactor[level];

        // Add border to coordinates and scale information
        const int nkps = keypoints.size();
        for(int i=0; i<nkps ; i++)
        {
            keypoints[i].pt.x+=minBorderX;
            keypoints[i].pt.y+=minBorderY;
            keypoints[i].octave=level;
            keypoints[i].size = scaledPatchSize;
        }
    }

    // compute orientations
    for (int level = 0; level < nlevels; ++level)
        computeOrientation(mvImagePyramid[level], allKeypoints[level], umax);
}

void ORBextractor::ComputeKeyPointsOld(std::vector<std::vector<KeyPoint> > &allKeypoints)
{
    allKeypoints.resize(nlevels);

    float imageRatio = (float)mvImagePyramid[0].cols/mvImagePyramid[0].rows;

    for (int level = 0; level < nlevels; ++level)
    {
        const int nDesiredFeatures = mnFeaturesPerLevel[level];

        const int levelCols = sqrt((float)nDesiredFeatures/(5*imageRatio));
        const int levelRows = imageRatio*levelCols;

        const int minBorderX = EDGE_THRESHOLD;
        const int minBorderY = minBorderX;
        const int maxBorderX = mvImagePyramid[level].cols-EDGE_THRESHOLD;
        const int maxBorderY = mvImagePyramid[level].rows-EDGE_THRESHOLD;

        const int W = maxBorderX - minBorderX;
        const int H = maxBorderY - minBorderY;
        const int cellW = ceil((float)W/levelCols);
        const int cellH = ceil((float)H/levelRows);

        const int nCells = levelRows*levelCols;
        const int nfeaturesCell = ceil((float)nDesiredFeatures/nCells);

        vector<vector<vector<KeyPoint> > > cellKeyPoints(levelRows, vector<vector<KeyPoint> >(levelCols));

        vector<vector<int> > nToRetain(levelRows,vector<int>(levelCols,0));
        vector<vector<int> > nTotal(levelRows,vector<int>(levelCols,0));
        vector<vector<bool> > bNoMore(levelRows,vector<bool>(levelCols,false));
        vector<int> iniXCol(levelCols);
        vector<int> iniYRow(levelRows);
        int nNoMore = 0;
        int nToDistribute = 0;


        float hY = cellH + 6;

        for(int i=0; i<levelRows; i++)
        {
            const float iniY = minBorderY + i*cellH - 3;
            iniYRow[i] = iniY;

            if(i == levelRows-1)
            {
                hY = maxBorderY+3-iniY;
                if(hY<=0)
                    continue;
            }

            float hX = cellW + 6;

            for(int j=0; j<levelCols; j++)
            {
                float iniX;

                if(i==0)
                {
                    iniX = minBorderX + j*cellW - 3;
                    iniXCol[j] = iniX;
                }
                else
                {
                    iniX = iniXCol[j];
                }


                if(j == levelCols-1)
                {
                    hX = maxBorderX+3-iniX;
                    if(hX<=0)
                        continue;
                }


                Mat cellImage = mvImagePyramid[level].rowRange(iniY,iniY+hY).colRange(iniX,iniX+hX);

                cellKeyPoints[i][j].reserve(nfeaturesCell*5);

                FAST(cellImage,cellKeyPoints[i][j],iniThFAST,true);

                if(cellKeyPoints[i][j].size()<=3)
                {
                    cellKeyPoints[i][j].clear();

                    FAST(cellImage,cellKeyPoints[i][j],minThFAST,true);
                }


                const int nKeys = cellKeyPoints[i][j].size();
                nTotal[i][j] = nKeys;

                if(nKeys>nfeaturesCell)
                {
                    nToRetain[i][j] = nfeaturesCell;
                    bNoMore[i][j] = false;
                }
                else
                {
                    nToRetain[i][j] = nKeys;
                    nToDistribute += nfeaturesCell-nKeys;
                    bNoMore[i][j] = true;
                    nNoMore++;
                }

            }
        }


        // Retain by score

        while(nToDistribute>0 && nNoMore<nCells)
        {
            int nNewFeaturesCell = nfeaturesCell + ceil((float)nToDistribute/(nCells-nNoMore));
            nToDistribute = 0;

            for(int i=0; i<levelRows; i++)
            {
                for(int j=0; j<levelCols; j++)
                {
                    if(!bNoMore[i][j])
                    {
                        if(nTotal[i][j]>nNewFeaturesCell)
                        {
                            nToRetain[i][j] = nNewFeaturesCell;
                            bNoMore[i][j] = false;
                        }
                        else
                        {
                            nToRetain[i][j] = nTotal[i][j];
                            nToDistribute += nNewFeaturesCell-nTotal[i][j];
                            bNoMore[i][j] = true;
                            nNoMore++;
                        }
                    }
                }
            }
        }

        vector<KeyPoint> & keypoints = allKeypoints[level];
        keypoints.reserve(nDesiredFeatures*2);

        const int scaledPatchSize = PATCH_SIZE*mvScaleFactor[level];

        // Retain by score and transform coordinates
        for(int i=0; i<levelRows; i++)
        {
            for(int j=0; j<levelCols; j++)
            {
                vector<KeyPoint> &keysCell = cellKeyPoints[i][j];
                KeyPointsFilter::retainBest(keysCell,nToRetain[i][j]);
                if((int)keysCell.size()>nToRetain[i][j])
                    keysCell.resize(nToRetain[i][j]);


                for(size_t k=0, kend=keysCell.size(); k<kend; k++)
                {
                    keysCell[k].pt.x+=iniXCol[j];
                    keysCell[k].pt.y+=iniYRow[i];
                    keysCell[k].octave=level;
                    keysCell[k].size = scaledPatchSize;
                    keypoints.push_back(keysCell[k]);
                }
            }
        }

        if((int)keypoints.size()>nDesiredFeatures)
        {
            KeyPointsFilter::retainBest(keypoints,nDesiredFeatures);
            keypoints.resize(nDesiredFeatures);
        }
    }

    // and compute orientations
    for (int level = 0; level < nlevels; ++level)
        computeOrientation(mvImagePyramid[level], allKeypoints[level], umax);
}


static void computeDescriptors(const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors,
                               const vector<Point>& pattern)
{
    descriptors = Mat::zeros((int)keypoints.size(), 32, CV_8UC1);

    for (size_t i = 0; i < keypoints.size(); i++)
        computeOrbDescriptor(keypoints[i], image, &pattern[0], descriptors.ptr((int)i));
}


/**
 * @brief
 * 通过调用 重载()运算符 来提取图像的orb的关键点和描述子
 * 
 * @param _image        获取的灰度图像
 * @param _mask         掩码
 * @param _keypoints    关键点位置
 * @param _descriptors  描述子
 * 括号运算符输入图像，并且传入引用参数_keypoints,_descriptors用于计算得到的特征点及其描述子
 */
void ORBextractor::operator()( InputArray _image, InputArray _mask, vector<KeyPoint>& _keypoints,
                      OutputArray _descriptors)
{ 
    if(_image.empty())
        return;

    Mat image = _image.getMat();
    assert(image.type() == CV_8UC1 );           // 判断通道是否为灰度图

    // Pre-compute the scale pyramid
    ComputePyramid(image);

    vector < vector<KeyPoint> > allKeypoints;   // 提取图像金字塔中各层图像的关键点并储存在allKeypoints里
    ComputeKeyPointsOctTree(allKeypoints);
    //ComputeKeyPointsOld(allKeypoints);

    // 新建descriptors对象，特征点的描述子将存在这里
    Mat descriptors;
    int nkeypoints = 0;                                 // 实际提取出来的关键点数量nkeypoints
    for (int level = 0; level < nlevels; ++level)
        nkeypoints += (int)allKeypoints[level].size();
    if( nkeypoints == 0 )
        _descriptors.release();
    else
    {
        // descriptors就好似nkeypoints*32维的矩阵，矩阵里存着一个uchar，其单位为8bit
        // 这样对于每个关键点，就对应有32*8=256bit的二进制向量作为描述子
        _descriptors.create(nkeypoints, 32, CV_8U);     // size
        descriptors = _descriptors.getMat();
    }

    _keypoints.clear();                                 // vector<KeyPoint>& _keypoints
    _keypoints.reserve(nkeypoints);

    int offset = 0;
    // 遍历高斯金字塔每层，计算其提取出的关键点的描述子储存在 descriptors 里
    for (int level = 0; level < nlevels; ++level)
    {
        vector<KeyPoint>& keypoints = allKeypoints[level];
        int nkeypointsLevel = (int)keypoints.size();

        if(nkeypointsLevel==0)
            continue;

        // preprocess the resized image
        Mat workingMat = mvImagePyramid[level].clone();
        // 使用高斯模糊
        GaussianBlur(workingMat, workingMat, Size(7, 7), 2, 2, BORDER_REFLECT_101);

        // Compute the descriptors
        // 计算描述子，其计算所需的点对分布采用的是高斯分布，储存在pattern里
        Mat desc = descriptors.rowRange(offset, offset + nkeypointsLevel);
        computeDescriptors(workingMat, keypoints, desc, pattern);

        offset += nkeypointsLevel;

        // Scale keypoint coordinates
        if (level != 0)
        {
            float scale = mvScaleFactor[level]; //getScale(level, firstLevel, scaleFactor);
            for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
                 keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint)
                keypoint->pt *= scale;          // 对关键点的位置坐做尺度恢复，恢复到原图的位置
        }
        // And add the keypoints to the output
        _keypoints.insert(_keypoints.end(), keypoints.begin(), keypoints.end());
    }
}


void ORBextractor::ComputePyramid(cv::Mat image)
{
    // 计算n个level尺度的图片
    for (int level = 0; level < nlevels; ++level)
    {
        float scale = mvInvScaleFactor[level];                                                      // 获取缩放尺度
        Size sz(cvRound((float)image.cols*scale), cvRound((float)image.rows*scale));                // 当前层图片尺寸
        Size wholeSize(sz.width + EDGE_THRESHOLD*2, sz.height + EDGE_THRESHOLD*2);                  // 截图前的图片尺寸
        Mat temp(wholeSize, image.type()), masktemp;                                                // 新建一个temp，大小为wholeSize

        /**
         * 从temp裁剪
         * 起点为EDGE_THRESHOLD, EDGE_THRESHOLD
         * 大小为sz.width, sz.height
         * 
         * 存入mvImagePyramid[level]
         */
        mvImagePyramid[level] = temp(Rect(EDGE_THRESHOLD, EDGE_THRESHOLD, sz.width, sz.height));

        // Compute the resized image
        /**
         * 生成mvImagePyramid[i]的图片的方法是：
         * 先用resize()将图片缩放至sz大小，放入 mvImagePyramid[i]
         * 接着用copyMakeBorder()扩展边界至wholeSize大小放入temp
         * 注意temp和mvImagePyramid[i]公用相同数据区，改变temp会改变mvImagePyramid[i]
         */
        if( level != 0 )
        {
            // 从上一级图像mvImagePyramid[level-1]中缩小图像至sz大小，并存入mvImagePyramid[level]
            resize(mvImagePyramid[level-1], mvImagePyramid[level], sz, 0, 0, INTER_LINEAR);

            // 扩充上下左右边界EDGE_THRESHOLD个像素，放入temp中，对mvImagePyramid无影响
            copyMakeBorder(mvImagePyramid[level], temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                           BORDER_REFLECT_101+BORDER_ISOLATED);            
        }
        else
        {
            // 需要扩充上下左右边界EDGE_THRESHOLD个像素，放入temp中，对mvImagePyramid无影响
            copyMakeBorder(image, temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                           BORDER_REFLECT_101);            
        }
    }

}

} //namespace ORB_SLAM
