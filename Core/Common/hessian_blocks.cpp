#include "Common/hessian_blocks.hpp"

namespace ds_slam
{

PointHessian::PointHessian(const ImmaturePoint *const rawPoint)
{
    shell = nullptr;

    instanceCounter++;
    host = rawPoint->host;
    hasDepthPrior = false;

    idepth_hessian = 0;
    maxRelBaseline = 0;
    numGoodResiduals = 0;

    // set static values & initialization.
    u = rawPoint->u;
    v = rawPoint->v;
    assert(std::isfinite(rawPoint->idepth_max));
    //idepth_init = rawPoint->idepth_GT;

    my_type = rawPoint->my_type;

    SetIdepthScaled((rawPoint->idepth_max + rawPoint->idepth_min) * 0.5);
    SetPointStatus(PointHessian::INACTIVE);

    int n = patternNum;
    memcpy(color, rawPoint->color, sizeof(float) * n);
    memcpy(weights, rawPoint->weights, sizeof(float) * n);
    energyTH = rawPoint->energyTH;

    efPoint = 0;
}

void PointHessian::Release()
{
    for (unsigned int i = 0; i < residuals.size(); i++)
        delete residuals[i];
    residuals.clear();
}

void FrameHessian::SetStateZero(const Vec10 &state_zero)
{
    assert(state_zero.head<6>().squaredNorm() < 1e-20);

    this->state_zero = state_zero;

    for (int i = 0; i < 6; i++)
    {
        Vec6 eps;
        eps.setZero();
        eps[i] = 1e-3;
        SE3 EepsP = Sophus::SE3::exp(eps);
        SE3 EepsM = Sophus::SE3::exp(-eps);
        SE3 w2c_leftEps_P_x0 = (GetWorldToCamEvalPT() * EepsP) * GetWorldToCamEvalPT().inverse();
        SE3 w2c_leftEps_M_x0 = (GetWorldToCamEvalPT() * EepsM) * GetWorldToCamEvalPT().inverse();
        nullspaces_pose.col(i) = (w2c_leftEps_P_x0.log() - w2c_leftEps_M_x0.log()) / (2e-3);
    }
    //nullspaces_pose.topRows<3>() *= SCALE_XI_TRANS_INVERSE;
    //nullspaces_pose.bottomRows<3>() *= SCALE_XI_ROT_INVERSE;

    // scale change
    SE3 w2c_leftEps_P_x0 = (GetWorldToCamEvalPT());
    w2c_leftEps_P_x0.translation() *= 1.00001;
    w2c_leftEps_P_x0 = w2c_leftEps_P_x0 * GetWorldToCamEvalPT().inverse();
    SE3 w2c_leftEps_M_x0 = (GetWorldToCamEvalPT());
    w2c_leftEps_M_x0.translation() /= 1.00001;
    w2c_leftEps_M_x0 = w2c_leftEps_M_x0 * GetWorldToCamEvalPT().inverse();
    nullspaces_scale = (w2c_leftEps_P_x0.log() - w2c_leftEps_M_x0.log()) / (2e-3);

    nullspaces_affine.setZero();
    nullspaces_affine.topLeftCorner<2, 1>() = Vec2(1, 0);
    assert(ab_exposure > 0);
    nullspaces_affine.topRightCorner<2, 1>() = Vec2(0, expf(AffG2LScaled().a) * ab_exposure);
};

void FrameHessian::Release()
{
    // DELETE POINT
    // DELETE RESIDUAL
    for (unsigned int i = 0; i < pointHessians.size(); i++)
        delete pointHessians[i];
    for (unsigned int i = 0; i < pointHessiansMarginalized.size(); i++)
        delete pointHessiansMarginalized[i];
    for (unsigned int i = 0; i < pointHessiansOut.size(); i++)
        delete pointHessiansOut[i];
    for (unsigned int i = 0; i < immaturePoints.size(); i++)
        delete immaturePoints[i];

    pointHessians.clear();
    pointHessiansMarginalized.clear();
    pointHessiansOut.clear();
    immaturePoints.clear();
}

void FrameHessian::MakeImages(float *color, CalibHessian *HCalib)
{

    for (int i = 0; i < pyrLevelsUsed; i++)
    {
        dIp[i] = new Eigen::Vector3f[wG[i] * hG[i]];
        absSquaredGrad[i] = new float[wG[i] * hG[i]];
    }
    dI = dIp[0]; // level = 0

    // make d0
    int w = wG[0];
    int h = hG[0];
    for (int i = 0; i < w * h; i++)
        dI[i][0] = color[i];

    for (int lvl = 0; lvl < pyrLevelsUsed; lvl++)
    {
        int wl = wG[lvl], hl = hG[lvl];
        Eigen::Vector3f *dI_l = dIp[lvl];

        float *dabs_l = absSquaredGrad[lvl];
        if (lvl > 0)
        {
            int lvlm1 = lvl - 1;
            int wlm1 = wG[lvlm1];
            Eigen::Vector3f *dI_lm = dIp[lvlm1];

            for (int y = 0; y < hl; y++)
            {
                for (int x = 0; x < wl; x++)
                {
                    dI_l[x + y * wl][0] = 0.25f * (dI_lm[2 * x + 2 * y * wlm1][0] +
                                                   dI_lm[2 * x + 1 + 2 * y * wlm1][0] +
                                                   dI_lm[2 * x + 2 * y * wlm1 + wlm1][0] +
                                                   dI_lm[2 * x + 1 + 2 * y * wlm1 + wlm1][0]); // 0 - value
                }
            }
        }

        for (int idx = wl; idx < wl * (hl - 1); idx++)
        {
            float dx = 0.5f * (dI_l[idx + 1][0] - dI_l[idx - 1][0]);
            float dy = 0.5f * (dI_l[idx + wl][0] - dI_l[idx - wl][0]);

            if (!std::isfinite(dx))
                dx = 0;
            if (!std::isfinite(dy))
                dy = 0;

            dI_l[idx][1] = dx; // 1 - dx
            dI_l[idx][2] = dy; // 2 - dy

            dabs_l[idx] = dx * dx + dy * dy;

            if (setting_gammaWeightsPixelSelect == 1 && HCalib != 0)
            {
                float gw = HCalib->GetBGradOnly((float)(dI_l[idx][0]));
                dabs_l[idx] *= gw * gw; // convert to gradient of original color space (before removing response).
            }
        }
    }
}

void FrameFramePrecalc::Set(FrameHessian *host, FrameHessian *target, CalibHessian *HCalib)
{
    this->host = host;
    this->target = target;

    SE3 leftToLeft_0 = target->GetWorldToCamEvalPT() * host->GetWorldToCamEvalPT().inverse();
    PRE_RTll_0 = (leftToLeft_0.rotationMatrix()).cast<float>();
    PRE_tTll_0 = (leftToLeft_0.translation()).cast<float>();

    SE3 leftToLeft = target->PRE_worldToCam * host->PRE_camToWorld;
    PRE_RTll = (leftToLeft.rotationMatrix()).cast<float>();
    PRE_tTll = (leftToLeft.translation()).cast<float>();
    distanceLL = leftToLeft.translation().norm();

    Mat33f K = Mat33f::Zero();
    K(0, 0) = HCalib->fxl();
    K(1, 1) = HCalib->fyl();
    K(0, 2) = HCalib->cxl();
    K(1, 2) = HCalib->cyl();
    K(2, 2) = 1;
    PRE_KRKiTll = K * PRE_RTll * K.inverse();
    PRE_RKiTll = PRE_RTll * K.inverse();
    PRE_KtTll = K * PRE_tTll;

    PRE_aff_mode = AffLight::FromToVecExposure(host->ab_exposure, target->ab_exposure, host->AffG2L(), target->AffG2L()).cast<float>();
    PRE_b0_mode = host->AffG2LScaled().b;
}

}