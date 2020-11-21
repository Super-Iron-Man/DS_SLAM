#pragma once

#include <vector>
#include <math.h>

#include "Utils/num_type.h"
#include "Optimization/raw_residual_jacobian.hpp"

namespace ds_slam
{

class PointFrameResidual;
class CalibHessian;
class FrameHessian;
class PointHessian;

class EFResidual;
class EFPoint;
class EFFrame;
class EnergyFunction;

class EFResidual
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    inline EFResidual(PointFrameResidual *org, EFPoint *point_, EFFrame *host_, EFFrame *target_) : data(org), point(point_), host(host_), target(target_)
    {
        isLinearized = false;
        isActiveAndIsGoodNEW = false;
        J = new RawResidualJacobian();
        assert(((long)this) % 16 == 0);
        assert(((long)J) % 16 == 0);
    }

    inline ~EFResidual()
    {
        delete J;
    }

    void TakeDataF();

    void FixLinearizationF(EnergyFunction *ef);

    // structural pointers
    PointFrameResidual *data;
    int hostIDX, targetIDX;
    EFPoint *point;
    EFFrame *host;
    EFFrame *target;
    int idxInAll;

    RawResidualJacobian *J;

    VecNRf res_toZeroF;
    Vec8f JpJdF;

    // status.
    bool isLinearized;

    // if residual is not OOB & not OUTLIER & should be used during accumulations
    bool isActiveAndIsGoodNEW;
    inline const bool &IsActive() const { return isActiveAndIsGoodNEW; }
};

enum EFPointStatus
{
    PS_GOOD = 0,
    PS_MARGINALIZE,
    PS_DROP
};

class EFPoint
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EFPoint(PointHessian *d, EFFrame *host_) : data(d), host(host_)
    {
        TakeData();
        stateFlag = EFPointStatus::PS_GOOD;
    }
    void TakeData();

    PointHessian *data;

    float priorF;
    float deltaF;

    // constant info (never changes in-between).
    int idxInPoints;
    EFFrame *host;

    // contains all residuals.
    std::vector<EFResidual *> residualsAll;

    float bdSumF;
    float HdiF;
    float Hdd_accLF;
    VecCf Hcd_accLF;
    float bd_accLF;
    float Hdd_accAF;
    VecCf Hcd_accAF;
    float bd_accAF;

    EFPointStatus stateFlag;
};

class EFFrame
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EFFrame(FrameHessian *d) : data(d)
    {
        TakeData();
    }
    void TakeData();

    Vec8 prior;       // prior hessian (diagonal)
    Vec8 delta_prior; // = state-state_prior (E_prior = (delta_prior)' * diag(prior) * (delta_prior)
    Vec8 delta;       // state - state_zero.

    std::vector<EFPoint *> points;
    FrameHessian *data;
    int idx; // idx in frames.

    int frameID;
};

}
