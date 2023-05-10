#pragma once
#include "DNDS_Macros.h"
#include "DNDS_Experimentals.h"
// #define NDEBUG
#include <assert.h>
#include <stdint.h>
#include <vector>
#include <memory>
#include <tuple>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <string>

static const std::string DNDS_Defines_state =
    std::string("DNDS_Defines ") + DNDS_Macros_State + DNDS_Experimentals_State
#ifdef NDEBUG
    + " NDEBUG "
#else
    + " (no NDEBUG) "
#endif
#ifdef NINSERT
    + " NINSERT "
#else
    + " (no NINSERT) "
#endif
    ;

/***************/ // DNDS_assertS

inline void DNDS_assert_false(const char *expr, const char *file, int line)
{
    std::cerr << "\033[91mDNDS DNDS_assertion failed\033[39m: \"" << expr << "\"  at [  " << file << ":" << line << "  ]" << std::endl;
    std::abort();
}

#ifdef DNDS_NDEBUG
#define DNDS_assert(expr)
#else
#define DNDS_assert(expr)\
       (static_cast<bool>(expr) \
            ? void(0)           \
            : DNDS_assert_false(#expr, __FILE__, __LINE__))
#endif

/***************/

static_assert(sizeof(uint8_t) == 1, "bad uint8_t");

namespace DNDS
{
    typedef double real;
    typedef int64_t index;
    typedef int32_t rowsize;

    static const char *outputDelim = "\t";

    typedef std::vector<rowsize> tRowsizeVec;
    typedef std::vector<index> tIndexVec;
    typedef std::shared_ptr<tIndexVec> tpIndexVec;

    typedef std::tuple<index, index> indexerPair;

    const index indexMin = INT64_MIN;

    const real UnInitReal = std::acos(-1) * 1e299 * std::sqrt(-1.0);

    const real veryLargeReal = 3e200;
    const real largeReal = 3e10;
    const real verySmallReal = 1e-200;
    const real smallReal = 1e-10;

    const real pi = std::acos(-1);

} // namespace DNDS

namespace DNDS
{

}

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
        for (typename TtIndexVec::size_type i = 1; i < rowstarts.size(); i++)
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

    /// \brief l must be non-negative, r must be positive. integers
    template <class TL, class TR>
    inline constexpr auto divCeil(TL l, TR r)
    {
        return l / r + (l % r) ? 1 : 0;
    }
}

/*









*/

namespace DNDS
{
    template <typename T>
    inline constexpr T sqr(const T &a)
    {
        return a * a;
    }

    inline constexpr real sign(real a)
    {
        return a > 0 ? 1 : (a < 0 ? -1 : 0);
    }

    inline constexpr real signP(real a)
    {
        return a >= 0 ? 1 : -1;
    }

    inline constexpr real signM(real a)
    {
        return a <= 0 ? -1 : 1;
    }

    template <typename T>
    inline constexpr T mod(T a, T b)
    {
        static_assert(std::is_signed<T>::value && std::is_integral<T>::value, "not legal mod type");
        T val = a % b;
        if (val < 0)
            val += b;
        return val;
    }
}

/*









*/

/*









*/

/*------------------------------------------*/
// Warning disabler:

#if defined(_MSC_VER)
#define DISABLE_WARNING_PUSH __pragma(warning(push))
#define DISABLE_WARNING_POP __pragma(warning(pop))
#define DISABLE_WARNING(warningNumber) __pragma(warning(disable \
                                                        : warningNumber))

#define DISABLE_WARNING_UNREFERENCED_FORMAL_PARAMETER DISABLE_WARNING(4100)
#define DISABLE_WARNING_UNREFERENCED_FUNCTION DISABLE_WARNING(4505)
// other warnings you want to deactivate...

#elif defined(__GNUC__) || defined(__clang__)
#define DO_PRAGMA(X) _Pragma(#X)
#define DISABLE_WARNING_PUSH DO_PRAGMA(GCC diagnostic push)
#define DISABLE_WARNING_POP DO_PRAGMA(GCC diagnostic pop)
#define DISABLE_WARNING(warningName) DO_PRAGMA(GCC diagnostic ignored warningName)

#define DISABLE_WARNING_UNREFERENCED_FORMAL_PARAMETER DISABLE_WARNING("-Wunused-parameter")
#define DISABLE_WARNING_UNREFERENCED_FUNCTION DISABLE_WARNING("-Wunused-function")
#define DISABLE_WARNING_DEPRECATED_DECLARATIONS DISABLE_WARNING("-Wdeprecated-declarations")
// other warnings you want to deactivate...

#else
#define DISABLE_WARNING_PUSH
#define DISABLE_WARNING_POP
#define DISABLE_WARNING_UNREFERENCED_FORMAL_PARAMETER
#define DISABLE_WARNING_UNREFERENCED_FUNCTION
// other warnings you want to deactivate...

#endif

/*------------------------------------------*/