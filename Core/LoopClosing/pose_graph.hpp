#pragma once

#include <thread>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <string>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <queue>
#include <assert.h>
#include <stdio.h>

#include "LoopClosing/keyframe.hpp"
#include "DBoW/DBoW2.h"
#include "DVision/DVision.h"
#include "DBoW/TemplatedDatabase.h"
#include "DBoW/TemplatedVocabulary.h"
#include "Common/frame.hpp"
#include "Visualizer/visualizer_3D.hpp"
#include "Utils/utility.hpp"



namespace ds_slam
{

class PoseGraph
{
public:
    //
    PoseGraph(std::vector<Frame *> *allKeyFramesHistory, std::vector<visualizer::Visualizer3D *> *viewers);

    //
    ~PoseGraph();

    //
    void LoadVocabulary(std::string voc_path);

    //
    void AddKeyFrame(KeyFrame *cur_kf, bool flag_detect_loop);

    //
    KeyFrame *GetKeyFrame(int index);

    //
    Frame *GetKeyFrameHistory(int index);

    //
    void SavePoseGraph();
    void LoadPoseGraph();

    // system drift
    Eigen::Vector3d t_drift;
    double yaw_drift;
    Eigen::Vector3d ypr_drift;
    Eigen::Matrix3d r_drift;
    double scale_drift;
    unsigned long latest_loop_index;

    // world frame(base sequence or first sequence)<----> cur sequence frame
    Eigen::Vector3d w_t_vio;
    Eigen::Matrix3d w_r_vio;

    // system members
    std::vector<Frame *> *allKeyFramesHistory;
    std::vector<visualizer::Visualizer3D *> *viewers;
private:
    // loop detection
    int DetectLoop(KeyFrame *keyframe, int frame_index);
    //
    void AddKeyFrameIntoVoc(KeyFrame *keyframe);
    // optimization function
    void OptimizePoseGraph();
    void Optimize7DoF();
    void Optimize4DoF();
    // update
    void UpdatePath();

    // keyframe list
    list<KeyFrame *> keyframelist;
    std::mutex m_keyframelist;
    std::mutex m_path;
    std::mutex m_drift;

    // optimization thread
    bool run_optimization;
    std::thread t_optimization;
    std::queue<int> optimize_buf;
    std::mutex m_optimize_buf;

    // sequence
    int sequence_cnt;
    vector<bool> sequence_loop;
    map<int, cv::Mat> image_pool;
    int earliest_loop_index;
    int base_sequence;

    // DBoW2 database and vocabulary
    BriefDatabase db;
    BriefVocabulary *voc;
    ORBDatabase db_orb;
    ORBVocabulary *voc_orb;
};

template <typename T>
T NormalizeAngle(const T &angle_degrees)
{
    if (angle_degrees > T(180.0))
        return angle_degrees - T(360.0);
    else if (angle_degrees < T(-180.0))
        return angle_degrees + T(360.0);
    else
        return angle_degrees;
};

class AngleLocalParameterization
{
public:
    template <typename T>
    bool operator()(const T *theta_radians, const T *delta_theta_radians,
                    T *theta_radians_plus_delta) const
    {
        *theta_radians_plus_delta =
            NormalizeAngle(*theta_radians + *delta_theta_radians);

        return true;
    }

    static ceres::LocalParameterization *Create()
    {
        return (new ceres::AutoDiffLocalParameterization<AngleLocalParameterization,
                                                         1, 1>);
    }
};

class AngleParameterization
{
public:
    template <typename T>
    bool operator()(const T *theta_radians, const T *delta_theta_radians,
                    T *theta_radians_plus_delta) const
    {
        theta_radians_plus_delta[0] = NormalizeAngle(theta_radians[0] + delta_theta_radians[0]);
        theta_radians_plus_delta[1] = NormalizeAngle(theta_radians[1] + delta_theta_radians[1]);
        theta_radians_plus_delta[2] = NormalizeAngle(theta_radians[2] + delta_theta_radians[2]);
        return true;
    }

    static ceres::LocalParameterization *Create()
    {
        return (new ceres::AutoDiffLocalParameterization<AngleLocalParameterization,
                                                         1, 1>);
    }
};

template <typename T>
void YawPitchRollToRotationMatrix(const T yaw, const T pitch, const T roll, T R[9])
{

    T y = yaw / T(180.0) * T(M_PI);
    T p = pitch / T(180.0) * T(M_PI);
    T r = roll / T(180.0) * T(M_PI);

    R[0] = cos(y) * cos(p);
    R[1] = -sin(y) * cos(r) + cos(y) * sin(p) * sin(r);
    R[2] = sin(y) * sin(r) + cos(y) * sin(p) * cos(r);
    R[3] = sin(y) * cos(p);
    R[4] = cos(y) * cos(r) + sin(y) * sin(p) * sin(r);
    R[5] = -cos(y) * sin(r) + sin(y) * sin(p) * cos(r);
    R[6] = -sin(p);
    R[7] = cos(p) * sin(r);
    R[8] = cos(p) * cos(r);
};

template <typename T>
void RotationMatrixTranspose(const T R[9], T inv_R[9])
{
    inv_R[0] = R[0];
    inv_R[1] = R[3];
    inv_R[2] = R[6];
    inv_R[3] = R[1];
    inv_R[4] = R[4];
    inv_R[5] = R[7];
    inv_R[6] = R[2];
    inv_R[7] = R[5];
    inv_R[8] = R[8];
};

template <typename T> 
void RotationMatrixRotatePoint(const T R[9], const T t[3], T r_t[3])
{
	r_t[0] = R[0] * t[0] + R[1] * t[1] + R[2] * t[2];
	r_t[1] = R[3] * t[0] + R[4] * t[1] + R[5] * t[2];
	r_t[2] = R[6] * t[0] + R[7] * t[1] + R[8] * t[2];
};

struct SevenDOFError
{
    SevenDOFError(double r_t_x, double r_t_y, double r_t_z, double r_yaw, double r_pitch, double r_roll, double r_scale)
        : r_t_x(r_t_x), r_t_y(r_t_y), r_t_z(r_t_z), r_yaw(r_yaw), r_pitch(r_pitch), r_roll(r_roll), r_scale(r_scale)
    {
    }

    template <typename T>
    bool operator()(const T *const ypr_i, const T *ti, const T *scale_i, const T *ypr_j, const T *tj, const T *scale_j, T *residuals) const
    {
        T t_w_ij[3];
        t_w_ij[0] = tj[0] - ti[0];
        t_w_ij[1] = tj[1] - ti[1];
        t_w_ij[2] = tj[2] - ti[2];

        // euler to rotation
        T w_R_i[9];
        YawPitchRollToRotationMatrix(ypr_i[0], ypr_i[1], ypr_i[2], w_R_i);
        // rotation transpose
        T i_R_w[9];
        RotationMatrixTranspose(w_R_i, i_R_w);
        // rotation matrix rotate point
        T t_i_ij[3];
        RotationMatrixRotatePoint(i_R_w, t_w_ij, t_i_ij);

        residuals[0] = (t_i_ij[0] - T(r_t_x));
        residuals[1] = (t_i_ij[1] - T(r_t_y));
        residuals[2] = (t_i_ij[2] - T(r_t_z));
        residuals[3] = NormalizeAngle(ypr_j[0] - ypr_i[0] - T(r_yaw));
        residuals[4] = NormalizeAngle(ypr_j[1] - ypr_i[1] - T(r_pitch));
        residuals[5] = NormalizeAngle(ypr_j[2] - ypr_i[2] - T(r_roll));
        residuals[6] = (scale_i[0] - scale_j[0] - T(r_scale));

        return true;
    }

    static ceres::CostFunction *Create(const double r_t_x, const double r_t_y, const double r_t_z,
                                       const double r_yaw, const double r_pitch, const double r_roll,
                                       const double r_scale)
    {
        return (new ceres::AutoDiffCostFunction<SevenDOFError, 7, 3, 3, 1, 3, 3, 1>(
            new SevenDOFError(r_t_x, r_t_y, r_t_z, r_yaw, r_pitch, r_roll, r_scale)));
    }

    double r_t_x, r_t_y, r_t_z;
    double r_yaw, r_pitch, r_roll;
    double r_scale;
};

struct SevenDOFWeightError
{
    SevenDOFWeightError(double r_t_x, double r_t_y, double r_t_z, double r_yaw, double r_pitch, double r_roll, double r_scale)
        : r_t_x(r_t_x), r_t_y(r_t_y), r_t_z(r_t_z), r_yaw(r_yaw), r_pitch(r_pitch), r_roll(r_roll), r_scale(r_scale)
    {
        weight = 1;
    }

    template <typename T>
    bool operator()(const T *const ypr_i, const T *ti, const T *scale_i, const T *ypr_j, const T *tj, const T *scale_j, T *residuals) const
    {
        T t_w_ij[3];
        t_w_ij[0] = tj[0] - ti[0];
        t_w_ij[1] = tj[1] - ti[1];
        t_w_ij[2] = tj[2] - ti[2];

        // euler to rotation
        T w_R_i[9];
        YawPitchRollToRotationMatrix(ypr_i[0], ypr_i[1], ypr_i[2], w_R_i);
        // rotation transpose
        T i_R_w[9];
        RotationMatrixTranspose(w_R_i, i_R_w);
        // rotation matrix rotate point
        T t_i_ij[3];
        RotationMatrixRotatePoint(i_R_w, t_w_ij, t_i_ij);

        residuals[0] = (t_i_ij[0] - T(r_t_x)) * T(weight);
        residuals[1] = (t_i_ij[1] - T(r_t_y)) * T(weight);
        residuals[2] = (t_i_ij[2] - T(r_t_z)) * T(weight);
        residuals[3] = NormalizeAngle(ypr_j[0] - ypr_i[0] - T(r_yaw)) * T(weight) / T(10.0);
        residuals[4] = NormalizeAngle(ypr_j[1] - ypr_i[1] - T(r_pitch)) * T(weight) / T(10.0);
        residuals[5] = NormalizeAngle(ypr_j[2] - ypr_i[2] - T(r_roll)) * T(weight) / T(10.0);
        residuals[6] = (scale_i[0] - scale_j[0] - T(r_scale)) * T(weight);

        return true;
    }

    static ceres::CostFunction *Create(const double r_t_x, const double r_t_y, const double r_t_z,
                                       const double r_yaw, const double r_pitch, const double r_roll,
                                       const double r_scale)
    {
        return (new ceres::AutoDiffCostFunction<SevenDOFWeightError, 7, 3, 3, 1, 3, 3, 1>(
            new SevenDOFWeightError(r_t_x, r_t_y, r_t_z, r_yaw, r_pitch, r_roll, r_scale)));
    }

    double r_t_x, r_t_y, r_t_z;
    double r_yaw, r_pitch, r_roll;
    double r_scale;
    double weight;
};

struct RelativePoseError
{
    RelativePoseError(Sim3 T_i_j)
    {
        Eigen::Vector3d t_i_j = T_i_j.translation();
        Eigen::Matrix3d R_i_j = T_i_j.rotationMatrix();
        Eigen::Vector3d ypr_i_j = Utility::R2ypr(R_i_j);
        double scale_i_j = T_i_j.scale();

        r_t_x = t_i_j[0]; r_t_y = t_i_j[1]; r_t_z = t_i_j[2];
        r_yaw = ypr_i_j[0]; r_pitch = ypr_i_j[1]; r_roll = ypr_i_j[2];
        r_scale = scale_i_j;
    }

    template <typename T>
    bool operator()(const T *const ypr_i, const T *ti, const T *scale_i, const T *ypr_j, const T *tj, const T *scale_j, T *residuals) const
    {
        T t_w_ij[3];
        t_w_ij[0] = tj[0] - ti[0];
        t_w_ij[1] = tj[1] - ti[1];
        t_w_ij[2] = tj[2] - ti[2];

        // euler to rotation
        T w_R_i[9];
        YawPitchRollToRotationMatrix(ypr_i[0], ypr_i[1], ypr_i[2], w_R_i);
        // rotation transpose
        T i_R_w[9];
        RotationMatrixTranspose(w_R_i, i_R_w);
        // rotation matrix rotate point
        T t_i_ij[3];
        RotationMatrixRotatePoint(i_R_w, t_w_ij, t_i_ij);

        residuals[0] = (t_i_ij[0] - T(r_t_x));
        residuals[1] = (t_i_ij[1] - T(r_t_y));
        residuals[2] = (t_i_ij[2] - T(r_t_z));
        residuals[3] = NormalizeAngle(ypr_j[0] - ypr_i[0] - T(r_yaw));
        residuals[4] = NormalizeAngle(ypr_j[1] - ypr_i[1] - T(r_pitch));
        residuals[5] = NormalizeAngle(ypr_j[2] - ypr_i[2] - T(r_roll));
        residuals[6] = (scale_i[0] - scale_j[0] - T(r_scale));

        return true;
    }

    static ceres::CostFunction *Create(const Sim3 T_i_j)
    {
        return (new ceres::AutoDiffCostFunction<RelativePoseError, 7, 3, 3, 1, 3, 3, 1>(
            new RelativePoseError(T_i_j)));
    }

    double r_t_x, r_t_y, r_t_z;
    double r_yaw, r_pitch, r_roll;
    double r_scale;
};

struct RelativePoseWeightError
{
    RelativePoseWeightError(Sim3 T_i_j)
    {
        Eigen::Vector3d t_i_j = T_i_j.translation();
        Eigen::Matrix3d R_i_j = T_i_j.rotationMatrix();
        Eigen::Vector3d ypr_i_j = Utility::R2ypr(R_i_j);
        double scale_i_j = T_i_j.scale();

        r_t_x = t_i_j[0]; r_t_y = t_i_j[1]; r_t_z = t_i_j[2];
        r_yaw = ypr_i_j[0]; r_pitch = ypr_i_j[1]; r_roll = ypr_i_j[2];
        r_scale = scale_i_j;
        weight = 1;
    }

    template <typename T>
    bool operator()(const T *const ypr_i, const T *ti, const T *scale_i, const T *ypr_j, const T *tj, const T *scale_j, T *residuals) const
    {
        T t_w_ij[3];
        t_w_ij[0] = tj[0] - ti[0];
        t_w_ij[1] = tj[1] - ti[1];
        t_w_ij[2] = tj[2] - ti[2];

        // euler to rotation
        T w_R_i[9];
        YawPitchRollToRotationMatrix(ypr_i[0], ypr_i[1], ypr_i[2], w_R_i);
        // rotation transpose
        T i_R_w[9];
        RotationMatrixTranspose(w_R_i, i_R_w);
        // rotation matrix rotate point
        T t_i_ij[3];
        RotationMatrixRotatePoint(i_R_w, t_w_ij, t_i_ij);

        residuals[0] = (t_i_ij[0] - T(r_t_x)) * T(weight);
        residuals[1] = (t_i_ij[1] - T(r_t_y)) * T(weight);
        residuals[2] = (t_i_ij[2] - T(r_t_z)) * T(weight);
        residuals[3] = NormalizeAngle(ypr_j[0] - ypr_i[0] - T(r_yaw)) * T(weight) / T(10.0);
        residuals[4] = NormalizeAngle(ypr_j[1] - ypr_i[1] - T(r_pitch)) * T(weight) / T(10.0);
        residuals[5] = NormalizeAngle(ypr_j[2] - ypr_i[2] - T(r_roll)) * T(weight) / T(10.0);
        residuals[6] = (scale_i[0] - scale_j[0] - T(r_scale)) * T(weight);

        return true;
    }

    static ceres::CostFunction *Create(const Sim3 T_i_j)
    {
        return (new ceres::AutoDiffCostFunction<RelativePoseWeightError, 7, 3, 3, 1, 3, 3, 1>(
            new RelativePoseWeightError(T_i_j)));
    }

    double r_t_x, r_t_y, r_t_z;
    double r_yaw, r_pitch, r_roll;
    double r_scale;
    double weight;
};

}