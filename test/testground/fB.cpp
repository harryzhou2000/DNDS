#include "../../DNDS_Profiling.h"

void *getB()
{
    return &DNDS::PerformanceTimer::Instance();
}