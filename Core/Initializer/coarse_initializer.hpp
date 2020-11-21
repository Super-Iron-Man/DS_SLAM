#ifndef __COARSE_INITIALIZER_HPP__
#define __COARSE_INITIALIZER_HPP__

#include <math.h>
#include <vector>

#include "Utils/num_type.h"
#include "Common/hessian_blocks.hpp"
#include "Optimization/matrix_accumulators.hpp"
#include "Visualizer/visualizer_3D.hpp"


namespace ds_slam
{

struct Pnt
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    // index in jacobian. never changes (actually, there is no reason why).
    float u, v;

    // idepth / isgood / energy during optimization.
    float idepth;
    bool isGood;
    Vec2f energy; // (UenergyPhotometric, energyRegularizer)
    bool isGood_new;
    float idepth_new;
    Vec2f energy_new;

    float iR;
    float iRSumNum;

    float lastHessian;
    float lastHessian_new;

    // max stepsize for idepth (corresponding to max. movement in pixel-space).
    float maxstep;

    // idx (x+y*w) of closest point one pyramid level above.
    int parent;
    float parentDist;

    // idx (x+y*w) of up to 10 nearest points in pixel space.
    int neighbours[10];
    float neighboursDist[10];

    float my_type;
    float outlierTH;
};

class CoarseInitializer
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    CoarseInitializer(int w, int h);
    ~CoarseInitializer();

    void SetFirst(CalibHessian *HCalib, FrameHessian *newFrameHessian);
    bool TrackFrame(FrameHessian *newFrameHessian, std::vector<visualizer::Visualizer3D *> &wraps);
    void CalcTGrads(FrameHessian *newFrameHessian);

    int frameID;
    bool fixAffine;
    bool printDebug;

    Pnt *points[PYR_LEVELS];
    int numPoints[PYR_LEVELS];
    AffLight thisToNext_aff;
    SE3 thisToNext;

    FrameHessian *firstFrame;
    FrameHessian *newFrame;

private:
    void MakeK(CalibHessian *HCalib);

    Vec3f CalcResAndGS(int lvl,
                       Mat88f &H_out, Vec8f &b_out,
                       Mat88f &H_out_sc, Vec8f &b_out_sc,
                       const SE3 &refToNew, AffLight refToNew_aff,
                       bool plot);
    Vec3f CalcEC(int lvl); // returns OLD NERGY, NEW ENERGY, NUM TERMS.
    void OptReg(int lvl);

    void PropagateUp(int srcLvl);
    void PropagateDown(int srcLvl);
    float Rescale();

    void ResetPoints(int lvl);
    void DoStep(int lvl, float lambda, Vec8f inc);
    void ApplyStep(int lvl);

    void MakeGradients(Eigen::Vector3f **data);

    void DebugPlot(int lvl, std::vector<visualizer::Visualizer3D *> &wraps);
    void MakeNN();

    // pyramid parameters
    Mat33 K[PYR_LEVELS];
    Mat33 Ki[PYR_LEVELS];
    double fx[PYR_LEVELS];
    double fy[PYR_LEVELS];
    double fxi[PYR_LEVELS];
    double fyi[PYR_LEVELS];
    double cx[PYR_LEVELS];
    double cy[PYR_LEVELS];
    double cxi[PYR_LEVELS];
    double cyi[PYR_LEVELS];
    int w[PYR_LEVELS];
    int h[PYR_LEVELS];

    bool snapped;
    int snappedAt;

    // pyramid images & levels on all levels
    Eigen::Vector3f *dINew[PYR_LEVELS];
    Eigen::Vector3f *dIFist[PYR_LEVELS];

    Eigen::DiagonalMatrix<float, 8> wM;

    // temporary buffers for H and b.
    Vec10f *JbBuffer; // 0-7: sum(dd * dp). 8: sum(res*dd). 9: 1/(1+sum(dd*dd))=inverse hessian entry.
    Vec10f *JbBuffer_new;

    Accumulator9 acc9;
    Accumulator9 acc9SC;

    Vec3f dGrads[PYR_LEVELS];

    float alphaK;
    float alphaW;
    float regWeight;
    float couplingWeight;
};

struct FLANNPointcloud
{
    inline FLANNPointcloud()
    {
        num = 0;
        points = 0;
    }

    inline FLANNPointcloud(int n, Pnt *p) : num(n), points(p) {}

    inline size_t KdtreeGetPointCount() const { return num; }

    inline float KdtreeDistance(const float *p1, const size_t idx_p2, size_t /*size*/) const
    {
        const float d0 = p1[0] - points[idx_p2].u;
        const float d1 = p1[1] - points[idx_p2].v;
        return d0 * d0 + d1 * d1;
    }

    inline float KdtreeGetPt(const size_t idx, int dim) const
    {
        if (dim == 0)
            return points[idx].u;
        else
            return points[idx].v;
    }
    template <class BBOX>
    bool KdtreeGetBbox(BBOX & /* bb */) const { return false; }

    int num;
    Pnt *points;
};
}


#endif