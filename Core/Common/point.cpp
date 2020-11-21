#include <assert.h>
#include "Common/point.hpp"
#include "Utils/global_calib.h"
#include "Common/hessian_blocks.hpp"


namespace ds_slam
{

static unsigned long pc_id = 0;

Point::Point(ImmaturePoint *rawPoint)
{
    host = rawPoint->host->shell;
    id = pc_id;
    uv[0] = rawPoint->u;
    uv[1] = rawPoint->v;
    idepth = (rawPoint->idepth_max + rawPoint->idepth_min) * 0.5;
    status = Point::PointStatus::VALID;

    //todo
    level = rawPoint->my_type;
    isCorner = rawPoint->isCorner;
    if(isCorner)
        assert(level == 0);

    pc_id++;
}

void Point::ComputeWorldPos()
{
    Sim3 Twc = host->TwcOpti;
#if 0
    Vec3 Kip = 1.0 / idepth * Vec3(fxiG[0] * uv[0] + cxiG[0],
                                   fyiG[0] * uv[1] + cyiG[0],
                                   1);
#else
    Vec3 Kip = 1.0 / idepth * Vec3(host->fxi * uv[0] + host->cxi,
                                   host->fyi * uv[1] + host->cyi,
                                   1);
#endif
    mWorldPos = Twc * Kip;
}

}