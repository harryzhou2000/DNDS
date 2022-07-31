#pragma once
#include "DNDS_Defines.h"
#include "DNDS_MPI.hpp"

namespace DNDS
{
    class PerformanceTimer // cxx11 + thread-safe singleton
    {
    public:
        enum TimerType
        {
            Unknown = 0,
            RHS = 1,
            Dt = 2,
            Reconstruction = 3,
            ReconstructionCR = 4,
            Limiter = 5,
            LimiterA = 6,
            LimiterB = 7,
            Basis = 8,
            Comm = 9,
        };

        static const int Ntype = 10;

    private:
        real timer[Ntype] = {0};
        real tStart[Ntype];
        PerformanceTimer(){};
        PerformanceTimer(const PerformanceTimer &);
        PerformanceTimer &operator=(const PerformanceTimer &);
    public:

        static PerformanceTimer& Instance();
        void StartTimer(TimerType t);
        void EndTimer(TimerType t);
        real getTimer(TimerType t);
        void clearTimer(TimerType t);
        void clearAllTimer();
        
    };

}