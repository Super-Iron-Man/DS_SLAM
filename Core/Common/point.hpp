#pragma once

#include "Utils/num_type.h"
#include "Common/immature_point.hpp"
#include "Common/frame.hpp"

namespace ds_slam
{

class Frame;

class Point
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    enum PointStatus
    {
        VALID,      // valid map point (but also outdated or margined)
        INVALID,    // invalid map point (outlier)
    };

    Point(ImmaturePoint *rawPoint);

    void ComputeWorldPos();

    // members
    Frame *host;    // the host frame
    unsigned long id = 0;   // id
    Vec3 mWorldPos = Vec3::Zero();  // pos in world
    Vec2f uv = Vec2f(0, 0); // pixel position in image
    float idepth = -1;        // inverse depth, invalid if < 0, computed by dso's sliding window
    PointStatus status;     // status

    float angle = 0;       // rotation
    float score = 0;       // shi-tomasi score
    bool isCorner = false; // indicating if this is a corner
    int level = 0;         // which pyramid level is the feature computed
};

}