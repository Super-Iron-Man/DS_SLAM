#pragma once

#include <vector>
#include <math.h>
 
#include "Utils/num_type.h"
#include "Utils/index_thread_reduce.hpp"
#include "Optimization/matrix_accumulators.hpp"


namespace ds_slam
{

class EFPoint;
class EnergyFunction;

class AccumulatedTopHessianSSE
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    inline AccumulatedTopHessianSSE()
    {
        for (int tid = 0; tid < NUM_THREADS; tid++)
        {
            nres[tid] = 0;
            acc[tid] = 0;
            nframes[tid] = 0;
        }
    };

    inline ~AccumulatedTopHessianSSE()
    {
        for (int tid = 0; tid < NUM_THREADS; tid++)
        {
            if (acc[tid] != 0)
                delete[] acc[tid];
        }
    };

    inline void SetZero(int nFrames, int min = 0, int max = 1, Vec10 *stats = 0, int tid = 0)
    {

        if (nFrames != nframes[tid])
        {
            if (acc[tid] != 0)
                delete[] acc[tid];
#if USE_XI_MODEL
            acc[tid] = new Accumulator14[nFrames * nFrames];
#else
            acc[tid] = new AccumulatorApprox[nFrames * nFrames];
#endif
        }

        for (int i = 0; i < nFrames * nFrames; i++)
        {
            acc[tid][i].Initialize();
        }

        nframes[tid] = nFrames;
        nres[tid] = 0;
    }

    void StitchDouble(MatXX &H, VecX &b, EnergyFunction const *const EF, bool usePrior, bool useDelta, int tid = 0);

    template <int mode>
    void AddPoint(EFPoint *p, EnergyFunction const *const ef, int tid = 0);

    void StitchDoubleMT(IndexThreadReduce<Vec10> *red, MatXX &H, VecX &b, EnergyFunction const *const EF, bool usePrior, bool MT)
    {
        // sum up, splitting by bock in square.
        if (MT)
        {
            MatXX Hs[NUM_THREADS];
            VecX bs[NUM_THREADS];
            for (int i = 0; i < NUM_THREADS; i++)
            {
                assert(nframes[0] == nframes[i]);
                Hs[i] = MatXX::Zero(nframes[0] * 8 + CPARS, nframes[0] * 8 + CPARS);
                bs[i] = VecX::Zero(nframes[0] * 8 + CPARS);
            }

            red->reduce(boost::bind(&AccumulatedTopHessianSSE::StitchDoubleInternal,
                                    this, Hs, bs, EF, usePrior, _1, _2, _3, _4),
                        0, nframes[0] * nframes[0], 0);

            // sum up results
            H = Hs[0];
            b = bs[0];

            for (int i = 1; i < NUM_THREADS; i++)
            {
                H.noalias() += Hs[i];
                b.noalias() += bs[i];
                nres[0] += nres[i];
            }
        }
        else
        {
            H = MatXX::Zero(nframes[0] * 8 + CPARS, nframes[0] * 8 + CPARS);
            b = VecX::Zero(nframes[0] * 8 + CPARS);
            StitchDoubleInternal(&H, &b, EF, usePrior, 0, nframes[0] * nframes[0], 0, -1);
        }

        // make diagonal by copying over parts.
        for (int h = 0; h < nframes[0]; h++)
        {
            int hIdx = CPARS + h * 8;
            H.block<CPARS, 8>(0, hIdx).noalias() = H.block<8, CPARS>(hIdx, 0).transpose();

            for (int t = h + 1; t < nframes[0]; t++)
            {
                int tIdx = CPARS + t * 8;
                H.block<8, 8>(hIdx, tIdx).noalias() += H.block<8, 8>(tIdx, hIdx).transpose();
                H.block<8, 8>(tIdx, hIdx).noalias() = H.block<8, 8>(hIdx, tIdx).transpose();
            }
        }
    }

    template <int mode>
    void AddPointsInternal(
        std::vector<EFPoint *> *points, EnergyFunction const *const ef,
        int min = 0, int max = 1, Vec10 *stats = 0, int tid = 0)
    {
        for (int i = min; i < max; i++)
            AddPoint<mode>((*points)[i], ef, tid);
    }

    int nframes[NUM_THREADS];
    EIGEN_ALIGN16 AccumulatorApprox *acc[NUM_THREADS];
    int nres[NUM_THREADS];

private:
    void StitchDoubleInternal(MatXX *H, VecX *b, EnergyFunction const *const EF, bool usePrior,
                              int min, int max, Vec10 *stats, int tid);
};

}
