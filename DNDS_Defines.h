#pragma once

// #define NDEBUG
#include <assert.h>
#include <stdint.h>
#include <vector>
#include <memory>
#include <tuple>

namespace DNDS
{
    typedef double real;
    typedef int64_t index;
    typedef int16_t rowsize;

    const char *outputDelim = "\t";

    typedef std::vector<rowsize> tRowsizeVec;
    typedef std::vector<index> tIndexVec;
    typedef std::shared_ptr<tIndexVec> tpIndexVec;
    
    typedef std::tuple<index,index> indexerPair;
    
} // namespace DNDS
