#pragma once

#include "Utils/num_type.h"
#include "Common/hessian_blocks.hpp"

namespace ds_slam
{

enum PixelSelectorStatus
{
    PIXSEL_VOID = 0,
    PIXSEL_1,
    PIXSEL_2,
    PIXSEL_3
};

class PixelSelector
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    PixelSelector(int w, int h);
    ~PixelSelector();

    int MakeMaps(const FrameHessian *const fh, int *map_out, int *corner_out, float density,
                 int recursionsLeft = 1, bool plot = false, float thFactor = 1);

    void MakeHists(const FrameHessian *const fh);

    int currentPotential;
    bool allowFast;

private:
    Eigen::Vector3i Select(const FrameHessian *const fh, int *map_out, int *corner_out,
                           int pot, float thFactor = 1);

    float ShiTomasiScore(const FrameHessian *fh, int u, int v, int halfbox = 4, int level = 0);

    unsigned char *randomPattern;

    int *gradHist;
    float *ths;
    float *thsSmoothed;
    int thsStep;
    const FrameHessian *gradHistFrame;
};

}