#pragma once
#include <math.h>
#include <vector>
#include <map>

#include "Utils/num_type.h"
#include "Utils/index_thread_reduce.hpp"
#include "Common/hessian_blocks.hpp"
#include "Optimization/energy_function_structs.hpp"
#include "Optimization/accumulated_top_hessian.hpp"
#include "Optimization/accumulated_SC_hessian.hpp"


namespace ds_slam
{

extern bool EFAdjointsValid;
extern bool EFIndicesValid;
extern bool EFDeltaValid;

class EnergyFunction
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    friend class EFFrame;
    friend class EFPoint;
    friend class EFResidual;
    friend class AccumulatedTopHessian;
    friend class AccumulatedTopHessianSSE;
    friend class AccumulatedSCHessian;
    friend class AccumulatedSCHessianSSE;

    EnergyFunction();
    ~EnergyFunction();

    EFResidual *InsertResidual(PointFrameResidual *r);
    EFFrame *InsertFrame(FrameHessian *fh, CalibHessian *Hcalib);
    EFPoint *InsertPoint(PointHessian *ph);

    void DropResidual(EFResidual *r);
    void MarginalizeFrame(EFFrame *fh);
    void RemovePoint(EFPoint *ph);

    void MarginalizePointsF();
    void DropPointsF();
    void SolveSystemF(int iteration, double lambda, CalibHessian *HCalib);
    double CalcMEnergyF();
    double calcLEnergyF_MT();

    void MakeIDX();

    void SetDeltaF(CalibHessian *HCalib);

    void SetAdjointsF(CalibHessian *Hcalib);

    std::vector<EFFrame *> frames;
    int nPoints, nFrames, nResiduals;

    MatXX HM;
    VecX bM;

    int resInA, resInL, resInM;
    MatXX lastHS;
    VecX lastbS;
    VecX lastX;
    std::vector<VecX> lastNullspaces_forLogging;
    std::vector<VecX> lastNullspaces_pose;
    std::vector<VecX> lastNullspaces_scale;
    std::vector<VecX> lastNullspaces_affA;
    std::vector<VecX> lastNullspaces_affB;

    IndexThreadReduce<Vec10> *red;

    std::map<uint64_t,
             Eigen::Vector2i,
             std::less<uint64_t>,
             Eigen::aligned_allocator<std::pair<const uint64_t, Eigen::Vector2i>>> connectivityMap;

private:
    VecX GetStitchedDeltaF() const;

    void ResubstituteF_MT(VecX x, CalibHessian *HCalib, bool MT);
    void ResubstituteFPt(const VecCf &xc, Mat18f *xAd, int min, int max, Vec10 *stats, int tid);

    void AccumulateAF_MT(MatXX &H, VecX &b, bool MT);
    void AccumulateLF_MT(MatXX &H, VecX &b, bool MT);
    void AccumulateSCF_MT(MatXX &H, VecX &b, bool MT);

    void CalcLEnergyPt(int min, int max, Vec10 *stats, int tid);

    void Orthogonalize(VecX *b, MatXX *H);
    
    Mat18f *adHTdeltaF;

    Mat88 *adHost;
    Mat88 *adTarget;

    Mat88f *adHostF;
    Mat88f *adTargetF;

    VecC cPrior;
    VecCf cDeltaF;
    VecCf cPriorF;

    AccumulatedTopHessianSSE *accSSE_top_L;
    AccumulatedTopHessianSSE *accSSE_top_A;

    AccumulatedSCHessianSSE *accSSE_bot;

    std::vector<EFPoint *> allPoints;
    std::vector<EFPoint *> allPointsToMarg;

    float currentLambda;
};

}

