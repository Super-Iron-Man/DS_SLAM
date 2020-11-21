#pragma once

#include <algorithm>
#include <vector>
#include <map>
#include <boost/thread.hpp>
#include <opencv2/opencv.hpp>
#include "Utils/num_type.h"
#include "Common/point.hpp"


namespace ds_slam
{

class Point;


class Frame
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    Frame();

    ~Frame();

    void RemoveOutlier(Point *outPoint);

    void CreateCvMat(Eigen::Vector3f *dI, int width, int height);

    // members
    int id;           // INTERNAL ID, starting at zero.
    int incoming_id;  // ID passed into DSO
    double timestamp; // timestamp passed into DSO.

    // set once after tracking
    SE3 camToTrackingRef;
    Frame *trackingRef;

    // constantly adapted.
    SE3 camToWorld; // Write: TRACKING, while frame is still fresh; MAPPING: only when locked [shellPoseMutex].
    AffLight AffG2L;
    bool poseValid;

    // statisitcs
    int statistics_outlierResOnThis;
    int statistics_goodResOnThis;
    int marginalizedAt;
    double movedByOpt;

    // points
    std::vector<Point *> points;

    // point cloud in window
    std::vector<Point *> window_PC;

    // intrinsic matrix
    float fx;
    float fy;
    float cx;
    float cy;
    float fxi;
    float fyi;
    float cxi;
    float cyi;

    // pose
    boost::mutex poseMutex;
    SE3 Twc;        // pose from camera to world, estimated by DSO
    Sim3 TwcOpti;   // pose from camera to world optimized by global pose graph

    // key frame
    bool is_kf;
    long index; //key frame id
    cv::Mat image;
    int sequence;
    long local_index; // local loop index

    // relative pose constraint between key-frames
    struct RELPOSE
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        RELPOSE(Sim3 T = Sim3(), const Mat77 &H = Mat77::Identity(), bool bIsLoop = false) : Trc(T), info(H), isLoop(bIsLoop) {}

        Sim3 Trc;                       // T_reference_current
        Mat77 info = Mat77::Identity(); // information matrix, inverse of covariance, default is identity
        bool isLoop = false;
    };

    // relative poses within the active window
    std::map<Frame *, RELPOSE, std::less<Frame *>, Eigen::aligned_allocator<std::pair<const Frame *, RELPOSE>>> poseRel;
    boost::mutex mutexPoseRel;
};

} // namespace ds_slam
