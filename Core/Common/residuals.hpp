#pragma once

#include <iostream>
#include <fstream>
#include <vector>

#include "Optimization/energy_function_structs.hpp"


namespace ds_slam
{

enum ResLocation
{
    ACTIVE = 0,
    LINEARIZED,
    MARGINALIZED,
    NONE
};

enum ResState
{
    IN = 0,
    OOB,
    OUTLIER
};

struct FullJacRowT
{
    Eigen::Vector2f projectedTo[MAX_RES_PER_POINT];
};


class FrameHessian;
class PointHessian;

class PointFrameResidual
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EFResidual *efResidual;

    static int instanceCounter;

    ResState state_state;
    double state_energy;
    ResState state_NewState;
    double state_NewEnergy;
    double state_NewEnergyWithOutlier;

    PointHessian *point;
    FrameHessian *host;
    FrameHessian *target;
    RawResidualJacobian *J;

    bool isNew;

    Eigen::Vector2f projectedTo[MAX_RES_PER_POINT];
    Vec3f centerProjectedTo;

    ~PointFrameResidual();
    PointFrameResidual();
    PointFrameResidual(PointHessian *point_, FrameHessian *host_, FrameHessian *target_);
    double Linearize(CalibHessian *HCalib);

    void SetState(ResState s) { state_state = s; }

    void ResetOOB()
    {
        state_NewEnergy = state_energy = 0;
        state_NewState = ResState::OUTLIER;

        SetState(ResState::IN);
    };
    void ApplyRes(bool copyJacobians);

    void DebugPlot();

    void PrintRows(std::vector<VecX> &v, VecX &r, int nFrames, int nPoints, int M, int res);
};
}

