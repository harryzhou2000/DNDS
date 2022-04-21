#include "DNDS_Defines.h"

namespace DNDS
{
    std::ostream *logStream;

    bool useCout = true;

    std::ostream &log() { return useCout ? std::cout : *logStream; }

    void setLogStream(std::ostream *nstream) { useCout = false, logStream = nstream; }
}