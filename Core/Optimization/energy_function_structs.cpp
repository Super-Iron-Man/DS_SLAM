#include "Optimization/energy_function_structs.hpp"
#include "Optimization/energy_function.hpp"
#include "Common/hessian_blocks.hpp"
#include "Common/residuals.hpp"


#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace ds_slam
{

void EFResidual::TakeDataF()
{
    std::swap<RawResidualJacobian *>(J, data->J);

    Vec2f JI_JI_Jd = J->JIdx2 * J->Jpdd;

    for (int i = 0; i < 6; i++)
        JpJdF[i] = J->Jpdxi[0][i] * JI_JI_Jd[0] + J->Jpdxi[1][i] * JI_JI_Jd[1];

    JpJdF.segment<2>(6) = J->JabJIdx * J->Jpdd;
}

void EFFrame::TakeData()
{
    prior = data->GetPrior().head<8>();
    delta = data->GetStateMinusStateZero().head<8>();
    delta_prior = (data->GetState() - data->GetPriorZero()).head<8>();

    //  Vec10 state_zero =  data->GetStateZero();
    //  state_zero.segment<3>(0) = SCALE_XI_TRANS * state_zero.segment<3>(0);
    //  state_zero.segment<3>(3) = SCALE_XI_ROT * state_zero.segment<3>(3);
    //  state_zero[6] = SCALE_A * state_zero[6];
    //  state_zero[7] = SCALE_B * state_zero[7];
    //  state_zero[8] = SCALE_A * state_zero[8];
    //  state_zero[9] = SCALE_B * state_zero[9];
    //
    //  std::cout << "state_zero: " << state_zero.transpose() << "\n";

    assert(data->frameID != -1);

    frameID = data->frameID;
}

void EFPoint::TakeData()
{
    priorF = data->hasDepthPrior ? setting_idepthFixPrior * SCALE_IDEPTH * SCALE_IDEPTH : 0;
    if (setting_solverMode & SOLVER_REMOVE_POSEPRIOR)
        priorF = 0;

    deltaF = data->idepth - data->idepth_zero;
}

void EFResidual::FixLinearizationF(EnergyFunction *ef)
{
    Vec8f dp = ef->adHTdeltaF[hostIDX + ef->nFrames * targetIDX];

    // compute Jp*delta
    __m128 Jp_delta_x = _mm_set1_ps(J->Jpdxi[0].dot(dp.head<6>()) + J->Jpdc[0].dot(ef->cDeltaF) + J->Jpdd[0] * point->deltaF);
    __m128 Jp_delta_y = _mm_set1_ps(J->Jpdxi[1].dot(dp.head<6>()) + J->Jpdc[1].dot(ef->cDeltaF) + J->Jpdd[1] * point->deltaF);
    __m128 delta_a = _mm_set1_ps((float)(dp[6]));
    __m128 delta_b = _mm_set1_ps((float)(dp[7]));

    for (int i = 0; i < patternNum; i += 4)
    {
        // PATTERN: rtz = resF - [JI*Jp Ja]*delta.
        __m128 rtz = _mm_load_ps(((float *)&J->resF) + i);
        rtz = _mm_sub_ps(rtz, _mm_mul_ps(_mm_load_ps(((float *)(J->JIdx)) + i), Jp_delta_x));
        rtz = _mm_sub_ps(rtz, _mm_mul_ps(_mm_load_ps(((float *)(J->JIdx + 1)) + i), Jp_delta_y));
        rtz = _mm_sub_ps(rtz, _mm_mul_ps(_mm_load_ps(((float *)(J->JabF)) + i), delta_a));
        rtz = _mm_sub_ps(rtz, _mm_mul_ps(_mm_load_ps(((float *)(J->JabF + 1)) + i), delta_b));
        _mm_store_ps(((float *)&res_toZeroF) + i, rtz);
    }

    isLinearized = true;
}

}
