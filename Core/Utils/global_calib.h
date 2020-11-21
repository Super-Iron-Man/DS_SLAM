#ifndef __GLOBAL_CALIB_H__
#define __GLOBAL_CALIB_H__


#include "Utils/settings.h"
#include "Utils/num_type.h"

namespace ds_slam
{
    extern int wG[PYR_LEVELS], hG[PYR_LEVELS];

    extern float fxG[PYR_LEVELS], fyG[PYR_LEVELS],
        cxG[PYR_LEVELS], cyG[PYR_LEVELS];

    extern float fxiG[PYR_LEVELS], fyiG[PYR_LEVELS],
        cxiG[PYR_LEVELS], cyiG[PYR_LEVELS];

    extern Eigen::Matrix3f KG[PYR_LEVELS], KiG[PYR_LEVELS];

    extern float wM3G;
    extern float hM3G;

    void SetGlobalCalib(int w, int h, const Eigen::Matrix3f &K);
} // namespace ds_slam

#endif