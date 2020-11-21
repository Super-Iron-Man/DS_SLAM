#pragma once

#include <vector>
#include <string>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include "Common/frame.hpp"
#include "DBoW/DBoW2.h"
#include "DVision/DVision.h"




namespace ds_slam
{

class KeyFrame
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    enum DescriptorType
    {
        BRIEF = 0,
        ORB,
    };
    DescriptorType extractor_type = DescriptorType::BRIEF;

    //
    KeyFrame(Frame *frame);

    //
    void ComputeWindowPointDescriptor();

    //
    void ComputePointDescriptor();

    //
    void ComputeWindowORBPoint();
    void ComputeORBPoint();

    //
    bool FindConnection(KeyFrame *old_kf);

    //
    int HammingDis(const DVision::BRIEF::bitset &a, const DVision::BRIEF::bitset &b);

    //
    bool SearchInAera(const DVision::BRIEF::bitset window_descriptor,
                      const std::vector<DVision::BRIEF::bitset> &descriptors_old,
                      const std::vector<cv::KeyPoint> &keypoints_old,
                      const std::vector<cv::KeyPoint> &keypoints_old_norm,
                      cv::Point2f &best_match,
                      cv::Point2f &best_match_norm);

    //
    void SearchByBRIEFDes(std::vector<cv::Point2f> &matched_2d_old,
                          std::vector<cv::Point2f> &matched_2d_old_norm,
                          std::vector<uchar> &status,
                          const std::vector<DVision::BRIEF::bitset> &descriptors_old,
                          const std::vector<cv::KeyPoint> &keypoints_old,
                          const std::vector<cv::KeyPoint> &keypoints_old_norm);

    //
    void FundmantalMatrixRANSAC(const std::vector<cv::Point2f> &matched_2d_cur_norm,
                                const std::vector<cv::Point2f> &matched_2d_old_norm,
                                vector<uchar> &status);

    //
    void PnPRANSAC(const vector<cv::Point2f> &matched_2d_old_norm,
                   const std::vector<cv::Point3f> &matched_3d,
                   std::vector<uchar> &status,
                   Eigen::Vector3d &PnP_T_old, Eigen::Matrix3d &PnP_R_old);

    //
    void GetVoPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i);

    //
    void GetPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i);

    //
    void GetScale(double &scale);

    //
    void UpdatePose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i);

    //
    void UpdateVoPose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i);

    //
    void UpdateScale(double &scale);

    //
    void UpdateLoop(Eigen::Matrix<double, 8, 1> &_loop_info);

    //
    Eigen::Vector3d GetLoopRelativeT();

    //
    double GetLoopRelativeYaw();

    //
    Eigen::Quaterniond GetLoopRelativeQ();

    // members
    Frame *frame;

    double time_stamp;
    long index; // global index
    long local_index; // local loop index

    Eigen::Vector3d vo_T_w_i; // window pose
    Eigen::Matrix3d vo_R_w_i;
    Eigen::Vector3d T_w_i; // optimized pose
    Eigen::Matrix3d R_w_i;
    Eigen::Vector3d origin_vo_T; // origin window pose
    Eigen::Matrix3d origin_vo_R;
    double vo_scale; // window scale
    double scale;

    cv::Mat image;
    cv::Mat thumbnail;

    std::vector<cv::Point3f> point_3d; // point cloud
    std::vector<cv::Point2f> point_2d_uv;
    std::vector<cv::Point2f> point_2d_norm;
    std::vector<double> point_id;
    std::vector<cv::KeyPoint> keypoints; // key point for loop detection
    std::vector<cv::KeyPoint> keypoints_norm;
    std::vector<cv::KeyPoint> window_keypoints; // key point by window
    std::vector<DVision::BRIEF::bitset> brief_descriptors; // brief descriptors
    std::vector<DVision::BRIEF::bitset> window_brief_descriptors; // brief descriptors by window
    std::vector<cv::Mat> orb_descriptors; // orb descriptors
    std::vector<cv::Mat> window_orb_descriptors;

    int sequence; // sequence differences cause relocation

    bool has_loop; // loop flag
    int loop_index;
    Eigen::Matrix<double, 8, 1> loop_info;

    // reserve for VIO
    Eigen::Vector3d tic;
    Eigen::Matrix3d qic;
};


}



