#pragma once

#include <string>
#include <vector>
#include <Eigen/Core>
#include "opencv2/opencv.hpp"
#include "Utils/global_calib.h"


namespace ds_slam
{

class ORBExtractor
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    const int HALF_PATCH_SIZE = 15; // half patch size for computing ORB descriptor

    ORBExtractor();

    void operator()(const cv::Mat &im, std::vector<cv::KeyPoint> &keys, std::vector<cv::Mat> &descriptors);


private:
    inline float IC_Angle(const cv::Mat &image, cv::Point2f pt, int level = 0)
    {
        float m_01 = 0, m_10 = 0;

        const uchar *center = &image.at<uchar>(cvRound(pt.y), cvRound(pt.x));

        // Treat the center line differently, v=0
        for (int u = -HALF_PATCH_SIZE; u <= HALF_PATCH_SIZE; ++u)
            m_10 += u * center[u];

        // Go line by line in the circular patch
        int step = (int)image.step1(); //wG[level];
        for (int v = 1; v <= HALF_PATCH_SIZE; ++v)
        {
            // Proceed over the two lines
            int v_sum = 0;
            int d = umax[v];
            for (int u = -d; u <= d; ++u)
            {
                int val_plus = center[u + v * step], val_minus = center[u - v * step];
                v_sum += (val_plus - val_minus);
                m_10 += u * (val_plus + val_minus);
            }
            m_01 += v * v_sum;
        }

        return atan2f(m_01, m_10);
    }

    // static data
    std::vector<int> umax;  // used to compute rotation
};

}