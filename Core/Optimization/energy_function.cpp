#include "Optimization/energy_function.hpp"


#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace ds_slam
{

bool EFAdjointsValid = false;
bool EFIndicesValid = false;
bool EFDeltaValid = false;

void EnergyFunction::SetAdjointsF(CalibHessian *Hcalib)
{

    if (adHost != 0)
        delete[] adHost;
    if (adTarget != 0)
        delete[] adTarget;
    adHost = new Mat88[nFrames * nFrames];
    adTarget = new Mat88[nFrames * nFrames];

    for (int h = 0; h < nFrames; h++)
        for (int t = 0; t < nFrames; t++)
        {
            FrameHessian *host = frames[h]->data;
            FrameHessian *target = frames[t]->data;

            SE3 hostToTarget = target->GetWorldToCamEvalPT() * host->GetWorldToCamEvalPT().inverse();

            Mat88 AH = Mat88::Identity();
            Mat88 AT = Mat88::Identity();

            AH.topLeftCorner<6, 6>() = -hostToTarget.Adj().transpose();
            AT.topLeftCorner<6, 6>() = Mat66::Identity();

            Vec2f affLL = AffLight::FromToVecExposure(host->ab_exposure, target->ab_exposure, host->AffG2LScaled(), target->AffG2LScaled()).cast<float>();
            AT(6, 6) = -affLL[0];
            AH(6, 6) = affLL[0];
            AT(7, 7) = -1;
            AH(7, 7) = affLL[0];

            AH.block<3, 8>(0, 0) *= SCALE_XI_TRANS;
            AH.block<3, 8>(3, 0) *= SCALE_XI_ROT;
            AH.block<1, 8>(6, 0) *= SCALE_A;
            AH.block<1, 8>(7, 0) *= SCALE_B;
            AT.block<3, 8>(0, 0) *= SCALE_XI_TRANS;
            AT.block<3, 8>(3, 0) *= SCALE_XI_ROT;
            AT.block<1, 8>(6, 0) *= SCALE_A;
            AT.block<1, 8>(7, 0) *= SCALE_B;

            adHost[h + t * nFrames] = AH;
            adTarget[h + t * nFrames] = AT;
        }
    cPrior = VecC::Constant(setting_initialCalibHessian);

    if (adHostF != 0)
        delete[] adHostF;
    if (adTargetF != 0)
        delete[] adTargetF;
    adHostF = new Mat88f[nFrames * nFrames];
    adTargetF = new Mat88f[nFrames * nFrames];

    for (int h = 0; h < nFrames; h++)
        for (int t = 0; t < nFrames; t++)
        {
            adHostF[h + t * nFrames] = adHost[h + t * nFrames].cast<float>();
            adTargetF[h + t * nFrames] = adTarget[h + t * nFrames].cast<float>();
        }

    cPriorF = cPrior.cast<float>();

    EFAdjointsValid = true;
}

EnergyFunction::EnergyFunction()
{
    adHost = 0;
    adTarget = 0;

    red = 0;

    adHostF = 0;
    adTargetF = 0;
    adHTdeltaF = 0;

    nFrames = nResiduals = nPoints = 0;

    HM = MatXX::Zero(CPARS, CPARS);
    bM = VecX::Zero(CPARS);

    accSSE_top_L = new AccumulatedTopHessianSSE();
    accSSE_top_A = new AccumulatedTopHessianSSE();
    accSSE_bot = new AccumulatedSCHessianSSE();

    resInA = resInL = resInM = 0;
    currentLambda = 0;
}

EnergyFunction::~EnergyFunction()
{
    for (EFFrame *f : frames)
    {
        for (EFPoint *p : f->points)
        {
            for (EFResidual *r : p->residualsAll)
            {
                r->data->efResidual = 0;
                delete r;
            }
            p->data->efPoint = 0;
            delete p;
        }
        f->data->efFrame = 0;
        delete f;
    }

    if (adHost != 0)
        delete[] adHost;
    if (adTarget != 0)
        delete[] adTarget;

    if (adHostF != 0)
        delete[] adHostF;
    if (adTargetF != 0)
        delete[] adTargetF;
    if (adHTdeltaF != 0)
        delete[] adHTdeltaF;

    delete accSSE_top_L;
    delete accSSE_top_A;
    delete accSSE_bot;
}

void EnergyFunction::SetDeltaF(CalibHessian *HCalib)
{
    if (adHTdeltaF != 0)
        delete[] adHTdeltaF;
    adHTdeltaF = new Mat18f[nFrames * nFrames];
    for (int h = 0; h < nFrames; h++)
        for (int t = 0; t < nFrames; t++)
        {
            int idx = h + t * nFrames;
            adHTdeltaF[idx] = frames[h]->data->GetStateMinusStateZero().head<8>().cast<float>().transpose() * adHostF[idx] +
                              frames[t]->data->GetStateMinusStateZero().head<8>().cast<float>().transpose() * adTargetF[idx];
        }

    cDeltaF = HCalib->value_minus_value_zero.cast<float>();
    for (EFFrame *f : frames)
    {
        f->delta = f->data->GetStateMinusStateZero().head<8>();
        f->delta_prior = (f->data->GetState() - f->data->GetPriorZero()).head<8>();

        for (EFPoint *p : f->points)
            p->deltaF = p->data->idepth - p->data->idepth_zero;
    }

    EFDeltaValid = true;
}

// accumulates & shifts L.
void EnergyFunction::AccumulateAF_MT(MatXX &H, VecX &b, bool MT)
{
    if (MT)
    {
        red->reduce(boost::bind(&AccumulatedTopHessianSSE::SetZero, accSSE_top_A, nFrames, _1, _2, _3, _4), 0, 0, 0);
        red->reduce(boost::bind(&AccumulatedTopHessianSSE::AddPointsInternal<0>,
                                accSSE_top_A, &allPoints, this, _1, _2, _3, _4),
                    0, allPoints.size(), 50);
        accSSE_top_A->StitchDoubleMT(red, H, b, this, false, true);
        resInA = accSSE_top_A->nres[0];
    }
    else
    {
        accSSE_top_A->SetZero(nFrames);
        for (EFFrame *f : frames)
            for (EFPoint *p : f->points)
                accSSE_top_A->AddPoint<0>(p, this);
        accSSE_top_A->StitchDoubleMT(red, H, b, this, false, false);
        resInA = accSSE_top_A->nres[0];
    }
}

// accumulates & shifts L.
void EnergyFunction::AccumulateLF_MT(MatXX &H, VecX &b, bool MT)
{
    if (MT)
    {
        red->reduce(boost::bind(&AccumulatedTopHessianSSE::SetZero, accSSE_top_L, nFrames, _1, _2, _3, _4), 0, 0, 0);
        red->reduce(boost::bind(&AccumulatedTopHessianSSE::AddPointsInternal<1>,
                                accSSE_top_L, &allPoints, this, _1, _2, _3, _4),
                    0, allPoints.size(), 50);
        accSSE_top_L->StitchDoubleMT(red, H, b, this, true, true);
        resInL = accSSE_top_L->nres[0];
    }
    else
    {
        accSSE_top_L->SetZero(nFrames);
        for (EFFrame *f : frames)
            for (EFPoint *p : f->points)
                accSSE_top_L->AddPoint<1>(p, this);
        accSSE_top_L->StitchDoubleMT(red, H, b, this, true, false);
        resInL = accSSE_top_L->nres[0];
    }
}

void EnergyFunction::AccumulateSCF_MT(MatXX &H, VecX &b, bool MT)
{
    if (MT)
    {
        red->reduce(boost::bind(&AccumulatedSCHessianSSE::SetZero, accSSE_bot, nFrames, _1, _2, _3, _4), 0, 0, 0);
        red->reduce(boost::bind(&AccumulatedSCHessianSSE::AddPointsInternal,
                                accSSE_bot, &allPoints, true, _1, _2, _3, _4),
                    0, allPoints.size(), 50);
        accSSE_bot->StitchDoubleMT(red, H, b, this, true);
    }
    else
    {
        accSSE_bot->SetZero(nFrames);
        for (EFFrame *f : frames)
            for (EFPoint *p : f->points)
                accSSE_bot->AddPoint(p, true);
        accSSE_bot->StitchDoubleMT(red, H, b, this, false);
    }
}

void EnergyFunction::ResubstituteF_MT(VecX x, CalibHessian *HCalib, bool MT)
{
    assert(x.size() == CPARS + nFrames * 8);

    VecXf xF = x.cast<float>();
    HCalib->step = -x.head<CPARS>();

    Mat18f *xAd = new Mat18f[nFrames * nFrames];
    VecCf cstep = xF.head<CPARS>();
    for (EFFrame *h : frames)
    {
        h->data->step.head<8>() = -x.segment<8>(CPARS + 8 * h->idx);
        h->data->step.tail<2>().setZero();

        for (EFFrame *t : frames)
            xAd[nFrames * h->idx + t->idx] = xF.segment<8>(CPARS + 8 * h->idx).transpose() * adHostF[h->idx + nFrames * t->idx] +
                                             xF.segment<8>(CPARS + 8 * t->idx).transpose() * adTargetF[h->idx + nFrames * t->idx];
    }

    if (MT)
        red->reduce(boost::bind(&EnergyFunction::ResubstituteFPt,
                                this, cstep, xAd, _1, _2, _3, _4),
                    0, allPoints.size(), 50);
    else
        ResubstituteFPt(cstep, xAd, 0, allPoints.size(), 0, 0);

    delete[] xAd;
}

void EnergyFunction::ResubstituteFPt(
    const VecCf &xc, Mat18f *xAd, int min, int max, Vec10 *stats, int tid)
{
    for (int k = min; k < max; k++)
    {
        EFPoint *p = allPoints[k];

        int ngoodres = 0;
        for (EFResidual *r : p->residualsAll)
            if (r->IsActive())
                ngoodres++;
        if (ngoodres == 0)
        {
            p->data->step = 0;
            continue;
        }
        float b = p->bdSumF;
        b -= xc.dot(p->Hcd_accAF + p->Hcd_accLF);

        for (EFResidual *r : p->residualsAll)
        {
            if (!r->IsActive())
                continue;
            b -= xAd[r->hostIDX * nFrames + r->targetIDX] * r->JpJdF;
        }

        p->data->step = -b * p->HdiF;
        assert(std::isfinite(p->data->step));
    }
}

double EnergyFunction::CalcMEnergyF()
{

    assert(EFDeltaValid);
    assert(EFAdjointsValid);
    assert(EFIndicesValid);

    VecX delta = GetStitchedDeltaF();
    return delta.dot(2 * bM + HM * delta);
}

void EnergyFunction::CalcLEnergyPt(int min, int max, Vec10 *stats, int tid)
{

    Accumulator11 E;
    E.Initialize();
    VecCf dc = cDeltaF;

    for (int i = min; i < max; i++)
    {
        EFPoint *p = allPoints[i];
        float dd = p->deltaF;

        for (EFResidual *r : p->residualsAll)
        {
            if (!r->isLinearized || !r->IsActive())
                continue;

            Mat18f dp = adHTdeltaF[r->hostIDX + nFrames * r->targetIDX];
            RawResidualJacobian *rJ = r->J;

            // compute Jp*delta
            float Jp_delta_x_1 = rJ->Jpdxi[0].dot(dp.head<6>()) + rJ->Jpdc[0].dot(dc) + rJ->Jpdd[0] * dd;

            float Jp_delta_y_1 = rJ->Jpdxi[1].dot(dp.head<6>()) + rJ->Jpdc[1].dot(dc) + rJ->Jpdd[1] * dd;

            __m128 Jp_delta_x = _mm_set1_ps(Jp_delta_x_1);
            __m128 Jp_delta_y = _mm_set1_ps(Jp_delta_y_1);
            __m128 delta_a = _mm_set1_ps((float)(dp[6]));
            __m128 delta_b = _mm_set1_ps((float)(dp[7]));

            for (int i = 0; i + 3 < patternNum; i += 4)
            {
                // PATTERN: E = (2*res_toZeroF + J*delta) * J*delta.
                __m128 Jdelta = _mm_mul_ps(_mm_load_ps(((float *)(rJ->JIdx)) + i), Jp_delta_x);
                Jdelta = _mm_add_ps(Jdelta, _mm_mul_ps(_mm_load_ps(((float *)(rJ->JIdx + 1)) + i), Jp_delta_y));
                Jdelta = _mm_add_ps(Jdelta, _mm_mul_ps(_mm_load_ps(((float *)(rJ->JabF)) + i), delta_a));
                Jdelta = _mm_add_ps(Jdelta, _mm_mul_ps(_mm_load_ps(((float *)(rJ->JabF + 1)) + i), delta_b));

                __m128 r0 = _mm_load_ps(((float *)&r->res_toZeroF) + i);
                r0 = _mm_add_ps(r0, r0);
                r0 = _mm_add_ps(r0, Jdelta);
                Jdelta = _mm_mul_ps(Jdelta, r0);
                E.UpdateSSENoShift(Jdelta);
            }
            for (int i = ((patternNum >> 2) << 2); i < patternNum; i++)
            {
                float Jdelta = rJ->JIdx[0][i] * Jp_delta_x_1 + rJ->JIdx[1][i] * Jp_delta_y_1 +
                               rJ->JabF[0][i] * dp[6] + rJ->JabF[1][i] * dp[7];
                E.UpdateSingleNoShift((float)(Jdelta * (Jdelta + 2 * r->res_toZeroF[i])));
            }
        }
        E.UpdateSingle(p->deltaF * p->deltaF * p->priorF);
    }
    E.Finish();
    (*stats)[0] += E.A;
}

double EnergyFunction::calcLEnergyF_MT()
{
    assert(EFDeltaValid);
    assert(EFAdjointsValid);
    assert(EFIndicesValid);

    double E = 0;
    for (EFFrame *f : frames)
        E += f->delta_prior.cwiseProduct(f->prior).dot(f->delta_prior);

    E += cDeltaF.cwiseProduct(cPriorF).dot(cDeltaF);

    red->reduce(boost::bind(&EnergyFunction::CalcLEnergyPt,
                            this, _1, _2, _3, _4),
                0, allPoints.size(), 50);

    return E + red->stats[0];
}

EFResidual *EnergyFunction::InsertResidual(PointFrameResidual *r)
{
    EFResidual *efr = new EFResidual(r, r->point->efPoint, r->host->efFrame, r->target->efFrame);
    efr->idxInAll = r->point->efPoint->residualsAll.size();
    r->point->efPoint->residualsAll.push_back(efr);

    connectivityMap[(((uint64_t)efr->host->frameID) << 32) + ((uint64_t)efr->target->frameID)][0]++;

    nResiduals++;
    r->efResidual = efr;
    return efr;
}

EFFrame *EnergyFunction::InsertFrame(FrameHessian *fh, CalibHessian *Hcalib)
{
    EFFrame *eff = new EFFrame(fh);
    eff->idx = frames.size();
    frames.push_back(eff);

    nFrames++;
    fh->efFrame = eff;

    assert(HM.cols() == 8 * nFrames + CPARS - 8);
    bM.conservativeResize(8 * nFrames + CPARS);
    HM.conservativeResize(8 * nFrames + CPARS, 8 * nFrames + CPARS);
    bM.tail<8>().setZero();
    HM.rightCols<8>().setZero();
    HM.bottomRows<8>().setZero();

    EFIndicesValid = false;
    EFAdjointsValid = false;
    EFDeltaValid = false;

    SetAdjointsF(Hcalib);
    MakeIDX();

    for (EFFrame *fh2 : frames)
    {
        connectivityMap[(((uint64_t)eff->frameID) << 32) + ((uint64_t)fh2->frameID)] = Eigen::Vector2i(0, 0);
        if (fh2 != eff)
            connectivityMap[(((uint64_t)fh2->frameID) << 32) + ((uint64_t)eff->frameID)] = Eigen::Vector2i(0, 0);
    }

    return eff;
}

EFPoint *EnergyFunction::InsertPoint(PointHessian *ph)
{
    EFPoint *efp = new EFPoint(ph, ph->host->efFrame);
    efp->idxInPoints = ph->host->efFrame->points.size();
    ph->host->efFrame->points.push_back(efp);

    nPoints++;
    ph->efPoint = efp;

    EFIndicesValid = false;

    return efp;
}

void EnergyFunction::DropResidual(EFResidual *r)
{
    EFPoint *p = r->point;
    assert(r == p->residualsAll[r->idxInAll]);

    p->residualsAll[r->idxInAll] = p->residualsAll.back();
    p->residualsAll[r->idxInAll]->idxInAll = r->idxInAll;
    p->residualsAll.pop_back();

    if (r->IsActive())
        r->host->data->shell->statistics_goodResOnThis++;
    else
        r->host->data->shell->statistics_outlierResOnThis++;

    connectivityMap[(((uint64_t)r->host->frameID) << 32) + ((uint64_t)r->target->frameID)][0]--;
    nResiduals--;
    r->data->efResidual = 0;
    delete r;
}

void EnergyFunction::MarginalizeFrame(EFFrame *fh)
{

    assert(EFDeltaValid);
    assert(EFAdjointsValid);
    assert(EFIndicesValid);

    assert((int)fh->points.size() == 0);
    int ndim = nFrames * 8 + CPARS - 8; // new dimension
    int odim = nFrames * 8 + CPARS;     // old dimension

    //	VecX eigenvaluesPre = HM.eigenvalues().real();
    //	std::sort(eigenvaluesPre.data(), eigenvaluesPre.data()+eigenvaluesPre.size());
    //

    if ((int)fh->idx != (int)frames.size() - 1)
    {
        int io = fh->idx * 8 + CPARS; // index of frame to move to end
        int ntail = 8 * (nFrames - fh->idx - 1);
        assert((io + 8 + ntail) == nFrames * 8 + CPARS);

        Vec8 bTmp = bM.segment<8>(io);
        VecX tailTMP = bM.tail(ntail);
        bM.segment(io, ntail) = tailTMP;
        bM.tail<8>() = bTmp;

        MatXX HtmpCol = HM.block(0, io, odim, 8);
        MatXX rightColsTmp = HM.rightCols(ntail);
        HM.block(0, io, odim, ntail) = rightColsTmp;
        HM.rightCols(8) = HtmpCol;

        MatXX HtmpRow = HM.block(io, 0, 8, odim);
        MatXX botRowsTmp = HM.bottomRows(ntail);
        HM.block(io, 0, ntail, odim) = botRowsTmp;
        HM.bottomRows(8) = HtmpRow;
    }

    //	// marginalize. First add prior here, instead of to active.
    HM.bottomRightCorner<8, 8>().diagonal() += fh->prior;
    bM.tail<8>() += fh->prior.cwiseProduct(fh->delta_prior);

    //	std::cout << std::setprecision(16) << "HMPre:\n" << HM << "\n\n";

    VecX SVec = (HM.diagonal().cwiseAbs() + VecX::Constant(HM.cols(), 10)).cwiseSqrt();
    VecX SVecI = SVec.cwiseInverse();

    //	std::cout << std::setprecision(16) << "SVec: " << SVec.transpose() << "\n\n";
    //	std::cout << std::setprecision(16) << "SVecI: " << SVecI.transpose() << "\n\n";

    // scale!
    MatXX HMScaled = SVecI.asDiagonal() * HM * SVecI.asDiagonal();
    VecX bMScaled = SVecI.asDiagonal() * bM;

    // invert bottom part!
    Mat88 hpi = HMScaled.bottomRightCorner<8, 8>();
    hpi = 0.5f * (hpi + hpi);
    hpi = hpi.inverse();
    hpi = 0.5f * (hpi + hpi);

    // schur-complement!
    MatXX bli = HMScaled.bottomLeftCorner(8, ndim).transpose() * hpi;
    HMScaled.topLeftCorner(ndim, ndim).noalias() -= bli * HMScaled.bottomLeftCorner(8, ndim);
    bMScaled.head(ndim).noalias() -= bli * bMScaled.tail<8>();

    //unscale!
    HMScaled = SVec.asDiagonal() * HMScaled * SVec.asDiagonal();
    bMScaled = SVec.asDiagonal() * bMScaled;

    // set.
    HM = 0.5 * (HMScaled.topLeftCorner(ndim, ndim) + HMScaled.topLeftCorner(ndim, ndim).transpose());
    bM = bMScaled.head(ndim);

    // remove from vector, without changing the order!
    for (unsigned int i = fh->idx; i + 1 < frames.size(); i++)
    {
        frames[i] = frames[i + 1];
        frames[i]->idx = i;
    }
    frames.pop_back();
    nFrames--;
    fh->data->efFrame = 0;

    assert((int)frames.size() * 8 + CPARS == (int)HM.rows());
    assert((int)frames.size() * 8 + CPARS == (int)HM.cols());
    assert((int)frames.size() * 8 + CPARS == (int)bM.size());
    assert((int)frames.size() == (int)nFrames);

    //	VecX eigenvaluesPost = HM.eigenvalues().real();
    //	std::sort(eigenvaluesPost.data(), eigenvaluesPost.data()+eigenvaluesPost.size());

    //	std::cout << std::setprecision(16) << "HMPost:\n" << HM << "\n\n";

    //	std::cout << "EigPre:: " << eigenvaluesPre.transpose() << "\n";
    //	std::cout << "EigPost: " << eigenvaluesPost.transpose() << "\n";

    EFIndicesValid = false;
    EFAdjointsValid = false;
    EFDeltaValid = false;

    MakeIDX();
    delete fh;
}

void EnergyFunction::MarginalizePointsF()
{
    assert(EFDeltaValid);
    assert(EFAdjointsValid);
    assert(EFIndicesValid);

    allPointsToMarg.clear();
    for (EFFrame *f : frames)
    {
        for (int i = 0; i < (int)f->points.size(); i++)
        {
            EFPoint *p = f->points[i];
            if (p->stateFlag == EFPointStatus::PS_MARGINALIZE)
            {
                p->priorF *= setting_idepthFixPriorMargFac;
                for (EFResidual *r : p->residualsAll)
                    if (r->IsActive())
                        connectivityMap[(((uint64_t)r->host->frameID) << 32) + ((uint64_t)r->target->frameID)][1]++;
                allPointsToMarg.push_back(p);
            }
        }
    }

    accSSE_bot->SetZero(nFrames);
    accSSE_top_A->SetZero(nFrames);
    for (EFPoint *p : allPointsToMarg)
    {
        accSSE_top_A->AddPoint<2>(p, this);
        accSSE_bot->AddPoint(p, false);
        RemovePoint(p);
    }
    MatXX M, Msc;
    VecX Mb, Mbsc;
    accSSE_top_A->StitchDouble(M, Mb, this, false, false);
    accSSE_bot->StitchDouble(Msc, Mbsc, this);

    resInM += accSSE_top_A->nres[0];

    MatXX H = M - Msc;
    VecX b = Mb - Mbsc;

    if (setting_solverMode & SOLVER_ORTHOGONALIZE_POINTMARG)
    {
        // have a look if prior is there.
        bool haveFirstFrame = false;
        for (EFFrame *f : frames)
            if (f->frameID == 0)
                haveFirstFrame = true;

        if (!haveFirstFrame)
            Orthogonalize(&b, &H);
    }

    HM += setting_margWeightFac * H;
    bM += setting_margWeightFac * b;

    if (setting_solverMode & SOLVER_ORTHOGONALIZE_FULL)
        Orthogonalize(&bM, &HM);

    EFIndicesValid = false;
    MakeIDX();
}

void EnergyFunction::DropPointsF()
{
    for (EFFrame *f : frames)
    {
        for (int i = 0; i < (int)f->points.size(); i++)
        {
            EFPoint *p = f->points[i];
            if (p->stateFlag == EFPointStatus::PS_DROP)
            {
                RemovePoint(p);
                i--;
            }
        }
    }

    EFIndicesValid = false;
    MakeIDX();
}

void EnergyFunction::RemovePoint(EFPoint *p)
{
    for (EFResidual *r : p->residualsAll)
        DropResidual(r);

    EFFrame *h = p->host;
    h->points[p->idxInPoints] = h->points.back();
    h->points[p->idxInPoints]->idxInPoints = p->idxInPoints;
    h->points.pop_back();

    nPoints--;
    p->data->efPoint = 0;

    EFIndicesValid = false;

    delete p;
}

void EnergyFunction::Orthogonalize(VecX *b, MatXX *H)
{
    //	VecX eigenvaluesPre = H.eigenvalues().real();
    //	std::sort(eigenvaluesPre.data(), eigenvaluesPre.data()+eigenvaluesPre.size());
    //	std::cout << "EigPre:: " << eigenvaluesPre.transpose() << "\n";

    // decide to which nullspaces to Orthogonalize.
    std::vector<VecX> ns;
    ns.insert(ns.end(), lastNullspaces_pose.begin(), lastNullspaces_pose.end());
    ns.insert(ns.end(), lastNullspaces_scale.begin(), lastNullspaces_scale.end());
    //	if(setting_affineOptModeA <= 0)
    //		ns.insert(ns.end(), lastNullspaces_affA.begin(), lastNullspaces_affA.end());
    //	if(setting_affineOptModeB <= 0)
    //		ns.insert(ns.end(), lastNullspaces_affB.begin(), lastNullspaces_affB.end());

    // make Nullspaces matrix
    MatXX N(ns[0].rows(), ns.size());
    for (unsigned int i = 0; i < ns.size(); i++)
        N.col(i) = ns[i].normalized();

    // compute Npi := N * (N' * N)^-1 = pseudo inverse of N.
    Eigen::JacobiSVD<MatXX> svdNN(N, Eigen::ComputeThinU | Eigen::ComputeThinV);

    VecX SNN = svdNN.singularValues();
    double minSv = 1e10, maxSv = 0;
    for (int i = 0; i < SNN.size(); i++)
    {
        if (SNN[i] < minSv)
            minSv = SNN[i];
        if (SNN[i] > maxSv)
            maxSv = SNN[i];
    }
    for (int i = 0; i < SNN.size(); i++)
    {
        if (SNN[i] > setting_solverModeDelta * maxSv)
            SNN[i] = 1.0 / SNN[i];
        else
            SNN[i] = 0;
    }

    MatXX Npi = svdNN.matrixU() * SNN.asDiagonal() * svdNN.matrixV().transpose(); // [dim] x 9.
    MatXX NNpiT = N * Npi.transpose();                                            // [dim] x [dim].
    MatXX NNpiTS = 0.5 * (NNpiT + NNpiT.transpose());                             // = N * (N' * N)^-1 * N'.

    if (b != 0)
        *b -= NNpiTS * *b;
    if (H != 0)
        *H -= NNpiTS * *H * NNpiTS;

    //	std::cout << std::setprecision(16) << "Orth SV: " << SNN.reverse().transpose() << "\n";

    //	VecX eigenvaluesPost = H.eigenvalues().real();
    //	std::sort(eigenvaluesPost.data(), eigenvaluesPost.data()+eigenvaluesPost.size());
    //	std::cout << "EigPost:: " << eigenvaluesPost.transpose() << "\n";
}

void EnergyFunction::SolveSystemF(int iteration, double lambda, CalibHessian *HCalib)
{
    if (setting_solverMode & SOLVER_USE_GN)
        lambda = 0;
    if (setting_solverMode & SOLVER_FIX_LAMBDA)
        lambda = 1e-5;

    assert(EFDeltaValid);
    assert(EFAdjointsValid);
    assert(EFIndicesValid);

    MatXX HL_top, HA_top, H_sc;
    VecX bL_top, bA_top, bM_top, b_sc;

    AccumulateAF_MT(HA_top, bA_top, multiThreading);

    AccumulateLF_MT(HL_top, bL_top, multiThreading);

    AccumulateSCF_MT(H_sc, b_sc, multiThreading);

    bM_top = (bM + HM * GetStitchedDeltaF());

    MatXX HFinal_top;
    VecX bFinal_top;

    if (setting_solverMode & SOLVER_ORTHOGONALIZE_SYSTEM)
    {
        // have a look if prior is there.
        bool haveFirstFrame = false;
        for (EFFrame *f : frames)
            if (f->frameID == 0)
                haveFirstFrame = true;

        MatXX HT_act = HL_top + HA_top - H_sc;
        VecX bT_act = bL_top + bA_top - b_sc;

        if (!haveFirstFrame)
            Orthogonalize(&bT_act, &HT_act);

        HFinal_top = HT_act + HM;
        bFinal_top = bT_act + bM_top;

        lastHS = HFinal_top;
        lastbS = bFinal_top;

        for (int i = 0; i < 8 * nFrames + CPARS; i++)
            HFinal_top(i, i) *= (1 + lambda);
    }
    else
    {

        HFinal_top = HL_top + HM + HA_top;
        bFinal_top = bL_top + bM_top + bA_top - b_sc;

        lastHS = HFinal_top - H_sc;
        lastbS = bFinal_top;

        for (int i = 0; i < 8 * nFrames + CPARS; i++)
            HFinal_top(i, i) *= (1 + lambda);
        HFinal_top -= H_sc * (1.0f / (1 + lambda));
    }

    VecX x;
    if (setting_solverMode & SOLVER_SVD)
    {
        VecX SVecI = HFinal_top.diagonal().cwiseSqrt().cwiseInverse();
        MatXX HFinalScaled = SVecI.asDiagonal() * HFinal_top * SVecI.asDiagonal();
        VecX bFinalScaled = SVecI.asDiagonal() * bFinal_top;
        Eigen::JacobiSVD<MatXX> svd(HFinalScaled, Eigen::ComputeThinU | Eigen::ComputeThinV);

        VecX S = svd.singularValues();
        double minSv = 1e10, maxSv = 0;
        for (int i = 0; i < S.size(); i++)
        {
            if (S[i] < minSv)
                minSv = S[i];
            if (S[i] > maxSv)
                maxSv = S[i];
        }

        VecX Ub = svd.matrixU().transpose() * bFinalScaled;
        int setZero = 0;
        for (int i = 0; i < Ub.size(); i++)
        {
            if (S[i] < setting_solverModeDelta * maxSv)
            {
                Ub[i] = 0;
                setZero++;
            }

            if ((setting_solverMode & SOLVER_SVD_CUT7) && (i >= Ub.size() - 7))
            {
                Ub[i] = 0;
                setZero++;
            }

            else
                Ub[i] /= S[i];
        }
        x = SVecI.asDiagonal() * svd.matrixV() * Ub;
    }
    else
    {
        VecX SVecI = (HFinal_top.diagonal() + VecX::Constant(HFinal_top.cols(), 10)).cwiseSqrt().cwiseInverse();
        MatXX HFinalScaled = SVecI.asDiagonal() * HFinal_top * SVecI.asDiagonal();
        x = SVecI.asDiagonal() * HFinalScaled.ldlt().solve(SVecI.asDiagonal() * bFinal_top); //  SVec.asDiagonal() * svd.matrixV() * Ub;
    }

    if ((setting_solverMode & SOLVER_ORTHOGONALIZE_X) || (iteration >= 2 && (setting_solverMode & SOLVER_ORTHOGONALIZE_X_LATER)))
    {
        VecX xOld = x;
        Orthogonalize(&x, 0);
    }

    lastX = x;

    //resubstituteF(x, HCalib);
    currentLambda = lambda;
    ResubstituteF_MT(x, HCalib, multiThreading);
    currentLambda = 0;
}

void EnergyFunction::MakeIDX()
{
    for (unsigned int idx = 0; idx < frames.size(); idx++)
        frames[idx]->idx = idx;

    allPoints.clear();

    for (EFFrame *f : frames)
        for (EFPoint *p : f->points)
        {
            allPoints.push_back(p);
            for (EFResidual *r : p->residualsAll)
            {
                r->hostIDX = r->host->idx;
                r->targetIDX = r->target->idx;
            }
        }

    EFIndicesValid = true;
}

VecX EnergyFunction::GetStitchedDeltaF() const
{
    VecX d = VecX(CPARS + nFrames * 8);
    d.head<CPARS>() = cDeltaF.cast<double>();
    for (int h = 0; h < nFrames; h++)
        d.segment<8>(CPARS + 8 * h) = frames[h]->delta;
    return d;
}

}
