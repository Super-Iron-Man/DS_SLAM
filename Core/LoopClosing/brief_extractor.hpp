#pragma once

#include <string>
#include <vector>
#include "opencv2/opencv.hpp"
#include "DVision/BRIEF.h"


namespace ds_slam
{

class BriefExtractor
{
public:
    BriefExtractor(const std::string &pattern_file);

    BriefExtractor();

    virtual void operator()(const cv::Mat &im, std::vector<cv::KeyPoint> &keys, std::vector<DVision::BRIEF::bitset> &descriptors) const;

    DVision::BRIEF m_brief;
};

}