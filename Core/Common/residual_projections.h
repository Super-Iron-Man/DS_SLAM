#pragma once

#include "Utils/num_type.h"
#include "Common/hessian_blocks.hpp"


namespace ds_slam
{

EIGEN_STRONG_INLINE float DeriveIdepth(const Vec3f &t, const float &u, const float &v,
                                       const int &dx, const int &dy, const float &dxInterp,
                                       const float &dyInterp, const float &drescale)
{
    return (dxInterp * drescale * (t[0] - t[2] * u) + dyInterp * drescale * (t[1] - t[2] * v)) * SCALE_IDEPTH;
}

EIGEN_STRONG_INLINE bool ProjectPoint(const float &u_pt, const float &v_pt,
                                      const float &idepth,
                                      const Mat33f &KRKi, const Vec3f &Kt,
                                      float &Ku, float &Kv)
{
    Vec3f ptp = KRKi * Vec3f(u_pt, v_pt, 1) + Kt * idepth;
    Ku = ptp[0] / ptp[2];
    Kv = ptp[1] / ptp[2];
    return Ku > 1.1f && Kv > 1.1f && Ku < wM3G && Kv < hM3G;
}

EIGEN_STRONG_INLINE bool ProjectPoint(const float &u_pt, const float &v_pt,
                                      const float &idepth,
                                      const int &dx, const int &dy,
                                      CalibHessian *const &HCalib,
                                      const Mat33f &R, const Vec3f &t,
                                      float &drescale, float &u, float &v,
                                      float &Ku, float &Kv, Vec3f &KliP, float &new_idepth)
{
    KliP = Vec3f((u_pt + dx - HCalib->cxl()) * HCalib->fxli(),
                 (v_pt + dy - HCalib->cyl()) * HCalib->fyli(),
                 1);

    Vec3f ptp = R * KliP + t * idepth;
    drescale = 1.0f / ptp[2];
    new_idepth = idepth * drescale;

    if (!(drescale > 0))
        return false;

    u = ptp[0] * drescale;
    v = ptp[1] * drescale;
    Ku = u * HCalib->fxl() + HCalib->cxl();
    Kv = v * HCalib->fyl() + HCalib->cyl();

    return Ku > 1.1f && Kv > 1.1f && Ku < wM3G && Kv < hM3G;
}

}
