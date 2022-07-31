#include "DNDS_Profiling.h"

namespace DNDS
{
    PerformanceTimer &PerformanceTimer::Instance()
    {
        static PerformanceTimer instance;
        return instance;
    }

    void PerformanceTimer::StartTimer(TimerType t)
    {
        tStart[t] = MPI_Wtime();
    }

    void PerformanceTimer::EndTimer(TimerType t)
    {
        this->timer[t] += MPI_Wtime() - tStart[t];
    }

    real PerformanceTimer::getTimer(TimerType t)
    {
        return timer[t];
    }

    void PerformanceTimer::clearTimer(TimerType t)
    {
        timer[t] = 0;
    }

    void PerformanceTimer::clearAllTimer()
    {
        for (int i = 0; i < Ntype; i++)
            clearTimer(TimerType(i));
    }
}