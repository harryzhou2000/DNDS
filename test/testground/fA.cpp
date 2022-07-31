#include "../../DNDS_Profiling.h"

void *getA()
{
    return &DNDS::PerformanceTimer::Instance();
}