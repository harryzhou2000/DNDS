#pragma once

// #define NDEBUG
#include <assert.h>
#include <stdint.h>
#include <vector>
#include <memory>
#include <tuple>
#include <iostream>
#include <cmath>
#include <iomanip>

namespace DNDS
{
    typedef double real;
    typedef int64_t index;
    typedef int16_t rowsize;

    static const char *outputDelim = "\t";

    typedef std::vector<rowsize> tRowsizeVec;
    typedef std::vector<index> tIndexVec;
    typedef std::shared_ptr<tIndexVec> tpIndexVec;

    typedef std::tuple<index, index> indexerPair;

    const index indexMin = INT64_MIN;

    const real UnInitReal = std::acos(-1) * 1e300;

    const real pi = std::acos(-1);

} // namespace DNDS

namespace DNDS
{
    extern std::ostream *logStream;

    extern bool useCout;

    std::ostream &log();

    void setLogStream(std::ostream *nstream);
}

/*









*/

namespace DNDS
{
    // Note that TtIndexVec being accumulated could overflow
    template <class TtRowsizeVec, class TtIndexVec>
    inline void AccumulateRowSize(const TtRowsizeVec &rowsizes, TtIndexVec &rowstarts)
    {
        rowstarts.resize(rowsizes.size() + 1);
        rowstarts[0] = 0;
        for (index i = 1; i < rowstarts.size(); i++)
            rowstarts[i] = rowstarts[i - 1] + rowsizes[i - 1];
    }

    template <class T>
    inline bool checkUniformVector(const std::vector<T> &dat, T &value)
    {
        if (dat.size() == 0)
            return false;
        value = dat[0];
        for (auto i = 1; i < dat.size(); i++)
            if (dat[i] != value)
                return false;
        return true;
    }

    template <class T, class TP = T>
    inline void PrintVec(const std::vector<T> &dat, std::ostream &out)
    {
        for (auto i = 0; i < dat.size(); i++)
            out << TP(dat[i]) << outputDelim;
    }
}

/*









*/