#ifndef __COARSE_TRACKER_HPP__
#define __COARSE_TRACKER_HPP__

#include <vector>
#include <math.h>

#include "Common/hessian_blocks.hpp"
#include "Utils/num_type.h"
#include "Utils/settings.h"
#include "Optimization/matrix_accumulators.hpp"
#include "Visualizer/visualizer_3D.hpp"



namespace ds_slam
{

class CoarseTracker
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    CoarseTracker(int w, int h);
    ~CoarseTracker();
    bool TrackNewestCoarse(FrameHessian *newFrameHessian,
                           SE3 &lastToNew_out, AffLight &aff_g2l_out,
                           int coarsestLvl, Vec5 minResForAbort,
                           visualizer::Visualizer3D *wrap = 0);
    void SetCoarseTrackingRef(std::vector<FrameHessian *> frameHessians);
    void MakeK(CalibHessian *HCalib);

    void DebugPlotIDepthMap(float *minID, float *maxID, std::vector<visualizer::Visualizer3D *> &wraps);
    void DebugPlotIDepthMapFloat(std::vector<visualizer::Visualizer3D *> &wraps);
    bool debugPrint, DebugPlot;

    Mat33f K[PYR_LEVELS];
    Mat33f Ki[PYR_LEVELS];
    float fx[PYR_LEVELS];
    float fy[PYR_LEVELS];
    float fxi[PYR_LEVELS];
    float fyi[PYR_LEVELS];
    float cx[PYR_LEVELS];
    float cy[PYR_LEVELS];
    float cxi[PYR_LEVELS];
    float cyi[PYR_LEVELS];
    int w[PYR_LEVELS];
    int h[PYR_LEVELS];

    FrameHessian *lastRef;
    AffLight lastRef_aff_g2l;
    FrameHessian *newFrame;
    int refFrameID;

    // act as pure ouptut
    Vec5 lastResiduals;
    Vec3 lastFlowIndicators;
    double firstCoarseRMSE;

private:
    void MakeCoarseDepthL0(std::vector<FrameHessian *> frameHessians);
    float *idepth[PYR_LEVELS];
    float *weightSums[PYR_LEVELS];
    float *weightSums_bak[PYR_LEVELS];

    Vec6 CalcResAndGS(int lvl, Mat88 &H_out, Vec8 &b_out, const SE3 &refToNew, AffLight AffG2L, float cutoffTH);
    Vec6 CalcRes(int lvl, const SE3 &refToNew, AffLight AffG2L, float cutoffTH);
    void CalcGSSSE(int lvl, Mat88 &H_out, Vec8 &b_out, const SE3 &refToNew, AffLight AffG2L);
    void CalcGS(int lvl, Mat88 &H_out, Vec8 &b_out, const SE3 &refToNew, AffLight AffG2L);

    // pc buffers
    float *pc_u[PYR_LEVELS];
    float *pc_v[PYR_LEVELS];
    float *pc_idepth[PYR_LEVELS];
    float *pc_color[PYR_LEVELS];
    int pc_n[PYR_LEVELS];

    // warped buffers
    float *buf_warped_idepth;
    float *buf_warped_u;
    float *buf_warped_v;
    float *buf_warped_dx;
    float *buf_warped_dy;
    float *buf_warped_residual;
    float *buf_warped_weight;
    float *buf_warped_refColor;
    int buf_warped_n;

    std::vector<float *> ptrToDelete;

    Accumulator9 acc;
};

class CoarseDistanceMap
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    CoarseDistanceMap(int w, int h);
    ~CoarseDistanceMap();

    void MakeDistanceMap(std::vector<FrameHessian *> frameHessians,
                         FrameHessian *frame);

    void MakeInlierVotes(std::vector<FrameHessian *> frameHessians);

    void MakeK(CalibHessian *HCalib);

    void AddIntoDistFinal(int u, int v);

    float *fwdWarpedIDDistFinal;

    Mat33f K[PYR_LEVELS];
    Mat33f Ki[PYR_LEVELS];
    float fx[PYR_LEVELS];
    float fy[PYR_LEVELS];
    float fxi[PYR_LEVELS];
    float fyi[PYR_LEVELS];
    float cx[PYR_LEVELS];
    float cy[PYR_LEVELS];
    float cxi[PYR_LEVELS];
    float cyi[PYR_LEVELS];
    int w[PYR_LEVELS];
    int h[PYR_LEVELS];

private:
    void GrowDistBFS(int bfsNum);

    PointFrameResidual **coarseProjectionGrid;
    int *coarseProjectionGridNum;
    Eigen::Vector2i *bfsList1;
    Eigen::Vector2i *bfsList2;
};

} // namespace ds_slam

#endif