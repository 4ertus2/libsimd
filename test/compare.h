#ifndef _SIMD_TEST_COMPARE_H_
#define _SIMD_TEST_COMPARE_H_

#include <limits>

struct Exception
{
    const char * func_;
    unsigned line_;
    unsigned length_;
};

#define FAIL() throw Exception({__PRETTY_FUNCTION__, __LINE__, length})

namespace
{
    template <typename T> T epsilon() { return 0; }
}

#if TEST_FIXED
#define TEST_FLOAT
namespace arithm
{
    using namespace ipp::f24;
    using namespace ipp::d53;
}
namespace
{
    #define EPS23 ( 1.19209289e-07f )
    #define EPS52 ( 2.2204460492503131e-016 )

    template <> float epsilon() { return EPS23; }
    template <> double epsilon() { return EPS52; }
}
#else
namespace arithm
{
    using namespace simd;
}
namespace
{
    template <> float epsilon() { return std::numeric_limits<float>::epsilon(); }
    template <> double epsilon() { return std::numeric_limits<double>::epsilon(); }
}
#endif

namespace
{
    template<typename T> bool equal(T x, T y) { return x == y; }

    template<> bool equal(float x, float y)
    {
        float r = x-y;
        if (r < 0)
            r = y-x;
        if (r < epsilon<float>())
            return true;
        return false;
    }

    template<> bool equal(double x, double y)
    {
        double r = x-y;
        if (r < 0)
            r = y-x;
        if (r < epsilon<double>())
            return true;
        return false;
    }
}

#endif
