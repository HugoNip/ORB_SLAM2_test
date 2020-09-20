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

#include "KeyFrameDatabase.h"

#include "KeyFrame.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"

#include<mutex>

using namespace std;

namespace ORB_SLAM2
{

KeyFrameDatabase::KeyFrameDatabase (const ORBVocabulary &voc):
    mpVoc(&voc)
{
    mvInvertedFile.resize(voc.size());
}



/**
 * @brief 
 * 根据关键帧的词包，更新数据库的倒排索引
 * 
 * 可以看到这个对象的实际作用是，可以通过图像上的特征点找到其他也包含相似特征点的**图像**
 * 这样进行重定位和回环检测的时候就不用在所有的图像中进行词袋向量的相似度查找了
 * 
 * @param pKF 关键帧
 */
void KeyFrameDatabase::add(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutex);

    for(DBoW2::BowVector::const_iterator vit= pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit!=vend; vit++)
        mvInvertedFile[vit->first].push_back(pKF);
}

void KeyFrameDatabase::erase(KeyFrame* pKF)
{
    unique_lock<mutex> lock(mMutex);

    // Erase elements in the Inverse File for the entry
    for(DBoW2::BowVector::const_iterator vit=pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit!=vend; vit++)
    {
        // List of keyframes that share the word
        list<KeyFrame*> &lKFs =   mvInvertedFile[vit->first];

        for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
        {
            if(pKF==*lit)
            {
                lKFs.erase(lit);
                break;
            }
        }
    }
}

void KeyFrameDatabase::clear()
{
    mvInvertedFile.clear();
    mvInvertedFile.resize(mpVoc->size());
}


/**
 * 在KeyFrameDatabase，以及与pKF在covisibility graph连接的keyframe中找出**与pKF可能形成闭环的候选帧**
 * 与DetectRelocalizationCandidates的区别是，在最开始先搜索了在covisibility graph与其连接的关键帧
 * 忽略和自己已有共视关系的关键帧
 * 
 * loop closing condition:
 * 1. covisible
 * 2. share words
 * 3. enough words
 * 
 * 在除去当前帧共视关系的关键帧数据中，检测闭环候选帧(这个函数在KeyFrameDatabase中)
 * 闭环候选帧删选过程：
 * 1. BoW得分>minScore;
 * 2. 统计满足1的关键帧中有共同单词最多的单词数maxcommonwords
 * 3. **筛选**出共同单词数大于mincommons(=0.8*maxcommons)的**关键帧**
 * 4. 相连的关键帧分为一组，计算组得分（总分）,得到最大总分bestAccScore,筛选出总分大于minScoreToRetain(=0.75*bestAccScore)的组
 *    用得分最高的候选帧IAccScoreAndMathch代表该组，计算组得分的目的是剔除单独一帧得分较高，但是没有共视关键帧作为闭环来说不够鲁棒
 *    对于通过了闭环检测的关键帧，还需要通过连续性检测(连续三帧都通过上面的筛选)，才能作为闭环候选帧
 */
vector<KeyFrame*> KeyFrameDatabase::DetectLoopCandidates(KeyFrame* pKF, float minScore)
{   
    // Part 1: sets contains KeyFrames that has a covisibile relationship with pKF
    // 返回此关键帧在Covisibility graph中与之相连接（有共视关系）的节点
    set<KeyFrame*> spConnectedKeyFrames = pKF->GetConnectedKeyFrames();
    // 和pKF有**相同单词**且具有**共视关系**的关键帧将会放在 lKFsSharingWords
    list<KeyFrame*> lKFsSharingWords;

    // Part 2: sets contains KeyFrames that shares same word pKF
    // Search all keyframes that **share a word** with current keyframes
    // Discard keyframes connected to the query keyframe
    // 找出与pKF有相同单词的关键帧
    {
        unique_lock<mutex> lock(mMutex);

        // 遍历pKF中Bow里的每个**单词**
        for(DBoW2::BowVector::const_iterator vit=pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit != vend; vit++)
        {
            // 通过单词id得到与之有相同单词的**关键帧**
            list<KeyFrame*> &lKFs = mvInvertedFile[vit->first];

            // 判断lKFs中关键帧是否复合要求
            for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
            {
                KeyFrame* pKFi=*lit;
                // 如果pKFi被标记访问过
                // 忽略和自己已有共视关系的关键帧
                if(pKFi->mnLoopQuery!=pKF->mnId)
                {
                    pKFi->mnLoopWords=0;
                    // 如果pKFi和pKF有共视关系
                    // 此处如果if条件成立，代表没有共视关系，此时才会进入执行语句
                    // 换言之，如果有共视关系，就直接忽略了，
                    // 这是它和DetectRelocalizationCandidates唯一的区别               
                    if(!spConnectedKeyFrames.count(pKFi))
                    {
                        pKFi->mnLoopQuery=pKF->mnId;
                        lKFsSharingWords.push_back(pKFi);
                    }
                }
                // 单词投票
                pKFi->mnLoopWords++;
            }
        }
    }

    // KFs from Part 1 and Part 2 are stored in lKFsSharingWords
    // 
    // ----------------------------- Finish searching candidates -------------------------------------
    // ----------------------------- Start compare ---------------------------------------------------

    if(lKFsSharingWords.empty())
        return vector<KeyFrame*>();

    list<pair<float,KeyFrame*> > lScoreAndMatch;

    // Only compare against those keyframes that share enough words
    // 在lKFsSharingWords找出nLoopWords的**最大值**，也就是**单词投票最多的关键帧**
    // also compute min value
    int maxCommonWords=0;
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        if((*lit)->mnLoopWords>maxCommonWords)
            maxCommonWords=(*lit)->mnLoopWords;
    }

    int minCommonWords = maxCommonWords*0.8f;

    int nscores=0;

    // select based on similarity score
    // Compute similarity score. Retain the matches whose score is higher than minScore
    // 遍历 lKFsSharingWords 中的keyframe，当其中的keyframe的 mLoopScore 大于阈值minScore则计算相似度后放入 lScoreAndMatch 中
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;

        if(pKFi->mnLoopWords>minCommonWords)    // need to have enough commonwords, then it can compute similarity score
        {
            nscores++;

            float si = mpVoc->score(pKF->mBowVec,pKFi->mBowVec);    // SImilarity score

            pKFi->mLoopScore = si;
            if(si>=minScore)
                lScoreAndMatch.push_back(make_pair(si,pKFi));       // list<pair<score, KF>>
        }
    }

    if(lScoreAndMatch.empty())
        return vector<KeyFrame*>();

    list<pair<float,KeyFrame*> > lAccScoreAndMatch;
    float bestAccScore = minScore;

    // ----------------------------------- Finish compute similarity score -------------------------------------
    // ----------------------------------- Start to select the best candidate ----------------------------------

    // select based on accumulate score
    // Lets now accumulate score by covisibility
    // 遍历 lScoreAndMatch 中的**keyframe**，找出其共视图中与此keyframe连接的权值前N=10的节点，加上原keyframe总共最多有**11个keyframe**
    // 累加这11个keyframe的相似度得分，然后在11个keyframe中选择相似度得分最高的那个放入 lAccScoreAndMatch 中
    // 在遍历过程中计算bestAccScore，也就是AccScore的最大值，后面的再次筛选有用
    for(list<pair<float,KeyFrame*> >::iterator it=lScoreAndMatch.begin(), itend=lScoreAndMatch.end(); it!=itend; it++)
    {
        KeyFrame* pKFi = it->second;
        // 返回共视图中与此keyframe连接的权值前10的节点keyframe
        vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);    // 11 KeyFrames

        float bestScore = it->first;
        float accScore = it->first;
        KeyFrame* pBestKF = pKFi;

        // for each keyframe 1/11
        for(vector<KeyFrame*>::iterator vit=vpNeighs.begin(), vend=vpNeighs.end(); vit!=vend; vit++)
        {
            KeyFrame* pKF2 = *vit;
            // 如果pKF2与pKF有相同的单词，且在essentialgraph中连接，
	        // 且pKF2的mLoopScore大于阈值minScore则将其相似度的值累加至accScore
            if(pKF2->mnLoopQuery==pKF->mnId && pKF2->mnLoopWords>minCommonWords)
            {
                accScore+=pKF2->mLoopScore;         // accumulate score
                if(pKF2->mLoopScore>bestScore)      // find a max value
                {
                    pBestKF=pKF2;                   // update BestKF to KF with best score
                    bestScore = pKF2->mLoopScore;   // update best score
                }
            }
        }

        lAccScoreAndMatch.push_back(make_pair(accScore,pBestKF));   // 11
        if(accScore>bestAccScore)                   // find a max value
            bestAccScore=accScore;                  // update best accscore
    }

    // Return all those keyframes with a score higher than 0.75*bestScore
    float minScoreToRetain = 0.75f*bestAccScore;

    set<KeyFrame*> spAlreadyAddedKF;
    vector<KeyFrame*> vpLoopCandidates;
    vpLoopCandidates.reserve(lAccScoreAndMatch.size()); // size: 11

    // 返回 lAccScoreAndMatch 中所有得分超过0.75*bestAccScore的keyframe集合
    for(list<pair<float,KeyFrame*> >::iterator it=lAccScoreAndMatch.begin(), itend=lAccScoreAndMatch.end(); it!=itend; it++)
    {
        if(it->first>minScoreToRetain)
        {
            KeyFrame* pKFi = it->second;
            if(!spAlreadyAddedKF.count(pKFi))
            {
                vpLoopCandidates.push_back(pKFi);
                spAlreadyAddedKF.insert(pKFi);
            }
        }
    }

    return vpLoopCandidates;    // vector<KeyFrame>, size: <= 11
}


/**
 * Detect Relocalization Candidates, return several candidates for selecting the best one
 * @return lAccScoreAndMatch 中所有得分超过 0.75*bestAccScore的keyframe 集合
 * 
 * 检测的主要步骤如下：
 * 1）找出 与当前帧 pKF 有公共单词的 所有关键帧 pKFi ，不包括 与当前帧相连的 关键帧。
 * 2）统计所有闭环候选帧中与 pKF 具有共同单词最多的单词数，只考虑共有单词数大于 
 *    0.8*maxCommonWords 以及匹配得分大于给定的 minScore 的关键帧，存入 lScoreAndMatch
 * 3）对于第二步中筛选出来的 pKFi ，每一个都要抽取出自身的共视（共享地图点最多的前10帧）关键帧分为一组，
 *    计算该组整体得分（与pKF比较的），记为bestAccScore。所有组得分大于 0.75*bestAccScore 的，
 *    均当作闭环候选帧。
 */
vector<KeyFrame*> KeyFrameDatabase::DetectRelocalizationCandidates(Frame *F)
{
    list<KeyFrame*> lKFsSharingWords;

    // Search all keyframes that share a word with current frame
    // 搜索 所有和 F 有着 相同单词 的 keyframe 存储在 lKFsSharingWords
    // 并且更新 keyframe 中 mnRelocWord s，表示和此F有多少共同的单词
    {
        unique_lock<mutex> lock(mMutex);

        // words 是检测图像是否匹配的枢纽，遍历该pKF的每一个word 
        for(DBoW2::BowVector::const_iterator vit=F->mBowVec.begin(), vend=F->mBowVec.end(); vit != vend; vit++)
        {
            // 提取所有包含该word的KeyFrame  
            list<KeyFrame*> &lKFs =   mvInvertedFile[vit->first];

            for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
            {
                KeyFrame* pKFi=*lit;
                if(pKFi->mnRelocQuery!=F->mnId)         // pKFi还没有标记为pKF的候选帧
                {
                    pKFi->mnRelocWords=0;
                    pKFi->mnRelocQuery=F->mnId;
                    lKFsSharingWords.push_back(pKFi);
                }
                pKFi->mnRelocWords++;
            }
        }
    }
    if(lKFsSharingWords.empty())
        return vector<KeyFrame*>();

    // Only compare against those keyframes that share enough words
    // 在 lKFsSharingWords 中，寻找 mnRelocWords 的最大值存入 maxCommonWords
    int maxCommonWords=0;
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        if((*lit)->mnRelocWords>maxCommonWords)
            maxCommonWords=(*lit)->mnRelocWords;
    }

    int minCommonWords = maxCommonWords*0.8f;

    list<pair<float,KeyFrame*> > lScoreAndMatch;

    int nscores=0;

    // Compute similarity score
    // 遍历 lKFsSharingWords 中的 keyframe ，当其中的 keyframe 的 mRelocScore 大于
    // 阈值 minCommonWords ,则计算相似度后放入 lScoreAndMatch 中
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;

        if(pKFi->mnRelocWords>minCommonWords)
        {
            nscores++;
            float si = mpVoc->score(F->mBowVec,pKFi->mBowVec);
            pKFi->mRelocScore=si;
            lScoreAndMatch.push_back(make_pair(si,pKFi));
        }
    }

    if(lScoreAndMatch.empty())
        return vector<KeyFrame*>();

    list<pair<float,KeyFrame*> > lAccScoreAndMatch;
    float bestAccScore = 0;

    // Lets now accumulate score by covisibility
    // 遍历 lScoreAndMatch 中的 keyframe，找出其共视图中与此 keyframe 连接的权值前N的节点，
    // 加上原 keyframe 总共11个 keyframe
    // 累加这11个keyframe的相似度得分，然后在11个keyframe中选择相似度得分最高的那个放入 lAccScoreAndMatch 中
    // 在遍历过程中计算 bestAccScore，也就是 AccScore 的最大值，后面的再次筛选有用
    for(list<pair<float,KeyFrame*> >::iterator it=lScoreAndMatch.begin(), itend=lScoreAndMatch.end(); it!=itend; it++)
    {
        KeyFrame* pKFi = it->second;
        // 返回共视图中与此 keyframe 连接的权值前10的节点keyframe
        vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

        float bestScore = it->first;
        float accScore = bestScore;
        KeyFrame* pBestKF = pKFi;
        for(vector<KeyFrame*>::iterator vit=vpNeighs.begin(), vend=vpNeighs.end(); vit!=vend; vit++)
        {
            KeyFrame* pKF2 = *vit;
            // 说明pKF2与F没有共同的单词，就放弃此循环的关键帧
            if(pKF2->mnRelocQuery!=F->mnId)
                continue;

            accScore+=pKF2->mRelocScore;
            if(pKF2->mRelocScore>bestScore)
            {
                pBestKF=pKF2;
                bestScore = pKF2->mRelocScore;
            }

        }
        lAccScoreAndMatch.push_back(make_pair(accScore,pBestKF));
        if(accScore>bestAccScore)
            bestAccScore=accScore;
    }

    // Return all those keyframes with a score higher than 0.75*bestScore
    // 返回 lAccScoreAndMatch 中所有得分超过 0.75*bestAccScore的keyframe 集合
    float minScoreToRetain = 0.75f*bestAccScore;
    set<KeyFrame*> spAlreadyAddedKF;
    vector<KeyFrame*> vpRelocCandidates;
    vpRelocCandidates.reserve(lAccScoreAndMatch.size());
    for(list<pair<float,KeyFrame*> >::iterator it=lAccScoreAndMatch.begin(), 
        itend=lAccScoreAndMatch.end(); it!=itend; it++)
    {
        const float &si = it->first;
        if(si>minScoreToRetain)
        {
            KeyFrame* pKFi = it->second;
            if(!spAlreadyAddedKF.count(pKFi))
            {
                vpRelocCandidates.push_back(pKFi);
                spAlreadyAddedKF.insert(pKFi);
            }
        }
    }

    return vpRelocCandidates;
}

} //namespace ORB_SLAM
