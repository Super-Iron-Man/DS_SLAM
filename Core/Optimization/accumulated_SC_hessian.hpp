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


class AccumulatedSCHessianSSE
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    inline AccumulatedSCHessianSSE()
    {
        for (int i = 0; i < NUM_THREADS; i++)
        {
            accE[i] = 0;
            accEB[i] = 0;
            accD[i] = 0;
            nframes[i] = 0;
        }
    };

    inline ~AccumulatedSCHessianSSE()
    {
        for (int i = 0; i < NUM_THREADS; i++)
        {
            if (accE[i] != 0)
                delete[] accE[i];
            if (accEB[i] != 0)
                delete[] accEB[i];
            if (accD[i] != 0)
                delete[] accD[i];
        }
    };

    inline void SetZero(int n, int min = 0, int max = 1, Vec10 *stats = 0, int tid = 0)
    {
        if (n != nframes[tid])
        {
            if (accE[tid] != 0)
                delete[] accE[tid];
            if (accEB[tid] != 0)
                delete[] accEB[tid];
            if (accD[tid] != 0)
                delete[] accD[tid];
            accE[tid] = new AccumulatorXX<8, CPARS>[n * n];
            accEB[tid] = new AccumulatorX<8>[n * n];
            accD[tid] = new AccumulatorXX<8, 8>[n * n * n];
        }
        accbc[tid].Initialize();
        accHcc[tid].Initialize();

        for (int i = 0; i < n * n; i++)
        {
            accE[tid][i].Initialize();
            accEB[tid][i].Initialize();

            for (int j = 0; j < n; j++)
                accD[tid][i * n + j].Initialize();
        }
        nframes[tid] = n;
    }

    void StitchDouble(MatXX &H_sc, VecX &b_sc, EnergyFunction const *const EF, int tid = 0);
    void AddPoint(EFPoint *p, bool shiftPriorToZero, int tid = 0);

    void StitchDoubleMT(IndexThreadReduce<Vec10> *red, MatXX &H, VecX &b, EnergyFunction const *const EF, bool MT)
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

            red->reduce(boost::bind(&AccumulatedSCHessianSSE::StitchDoubleInternal,
                                    this, Hs, bs, EF, _1, _2, _3, _4),
                        0, nframes[0] * nframes[0], 0);

            // sum up results
            H = Hs[0];
            b = bs[0];

            for (int i = 1; i < NUM_THREADS; i++)
            {
                H.noalias() += Hs[i];
                b.noalias() += bs[i];
            }
        }
        else
        {
            H = MatXX::Zero(nframes[0] * 8 + CPARS, nframes[0] * 8 + CPARS);
            b = VecX::Zero(nframes[0] * 8 + CPARS);
            StitchDoubleInternal(&H, &b, EF, 0, nframes[0] * nframes[0], 0, -1);
        }

        // make diagonal by copying over parts.
        for (int h = 0; h < nframes[0]; h++)
        {
            int hIdx = CPARS + h * 8;
            H.block<CPARS, 8>(0, hIdx).noalias() = H.block<8, CPARS>(hIdx, 0).transpose();
        }
    }

    void AddPointsInternal(std::vector<EFPoint *> *points, bool shiftPriorToZero,
                           int min = 0, int max = 1, Vec10 *stats = 0, int tid = 0)
    {
        for (int i = min; i < max; i++)
            AddPoint((*points)[i], shiftPriorToZero, tid);
    }

    AccumulatorXX<8, CPARS> *accE[NUM_THREADS];
    AccumulatorX<8> *accEB[NUM_THREADS];
    AccumulatorXX<8, 8> *accD[NUM_THREADS];
    AccumulatorXX<CPARS, CPARS> accHcc[NUM_THREADS];
    AccumulatorX<CPARS> accbc[NUM_THREADS];
    int nframes[NUM_THREADS];

private:
    void StitchDoubleInternal(MatXX *H, VecX *b, EnergyFunction const *const EF,
                              int min, int max, Vec10 *stats, int tid);
};

}

