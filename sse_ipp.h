#pragma once

#ifdef max
#undef max
#endif

#ifdef min
#undef min
#endif

#ifndef _SIMD_EXT_T
#define _SIMD_EXT_T template<typename _T> extern
#endif

#ifndef _SIMD_EXT_TU
#define _SIMD_EXT_TU template<typename _T, typename _U> extern
#endif

namespace ipp
{
    namespace common
    {
        void ipp_free(void* ptr);
        template<typename T> void free(T * ptr) { ipp_free((void*)ptr); }

        _SIMD_EXT_T _T* malloc(int len);

        _SIMD_EXT_T void zero(_T* pDst, int len);
        _SIMD_EXT_T void set(_T val, _T* pDst, int len);
        _SIMD_EXT_T void copy(const _T* pSrc, _T* pDst, int len);
        _SIMD_EXT_T void move(const _T* pSrc, _T* pDst, int len);

        _SIMD_EXT_TU void convert(const _T* pSrc, _U* pDst, int len);
    }

    namespace arithmetic
    {
        _SIMD_EXT_T void addC(const _T* pSrc, _T val, _T* pDst, int len);
        _SIMD_EXT_T void add(const _T* pSrc1, const _T* pSrc2, _T* pDst, int len);

        _SIMD_EXT_T void subC(const _T* pSrc, _T val, _T* pDst, int len);
        _SIMD_EXT_T void subCRev(const _T* pSrc, _T val, _T* pDst, int len);
        _SIMD_EXT_T void sub(const _T* pSrc1, const _T* pSrc2, _T* pDst, int len);

        _SIMD_EXT_T void mulC(const _T* pSrc, _T val, _T* pDst, int len);
        _SIMD_EXT_T void mul(const _T* pSrc1, const _T* pSrc2, _T* pDst, int len);

        _SIMD_EXT_T void divC(const _T* pSrc, _T val, _T* pDst, int len);
        _SIMD_EXT_T void divCRev(const _T* pSrc, _T val, _T* pDst, int len);
        _SIMD_EXT_T void div(const _T* pSrc1, const _T* pSrc2, _T* pDst, int len);

        _SIMD_EXT_T void abs(const _T* pSrc, _T* pDst, int len);

        namespace f21
        {
            void div(const float * pSrc1, const float * pSrc2, float * pDst, int len);
        }
        namespace f24
        {
            void add(const float * pSrc1, const float * pSrc2, float * pDst, int len);
            void sub(const float * pSrc1, const float * pSrc2, float * pDst, int len);
            void mul(const float * pSrc1, const float * pSrc2, float * pDst, int len);
            void div(const float * pSrc1, const float * pSrc2, float * pDst, int len);
            void abs(const float * pSrc, float * pDst, int len);
        }
        namespace d50
        {
            void div(const double * pSrc1, const double * pSrc2, double * pDst, int len);
        }
        namespace d53
        {
            void add(const double * pSrc1, const double * pSrc2, double * pDst, int len);
            void sub(const double * pSrc1, const double * pSrc2, double * pDst, int len);
            void mul(const double * pSrc1, const double * pSrc2, double * pDst, int len);
            void div(const double * pSrc1, const double * pSrc2, double * pDst, int len);
            void abs(const double * pSrc, double * pDst, int len);
        }
    }

    namespace power
    {
        _SIMD_EXT_T void inv(const _T* pSrc, _T* pDst, int len);
        _SIMD_EXT_T void sqr(const _T* pSrc, _T* pDst, int len);
        _SIMD_EXT_T void sqrt(const _T* pSrc, _T* pDst, int len);
        _SIMD_EXT_T void invSqrt(const _T* pSrc, _T* pDst, int len);
        _SIMD_EXT_T void powx(const _T* pSrc, const _T constValue, _T* pDst, int len);
        _SIMD_EXT_T void pow(const _T* pSrc1, const _T* pSrc2, _T* pDst, int len);
        _SIMD_EXT_T void cbrt(const _T* pSrc, _T* pDst, int len);
        _SIMD_EXT_T void hypot(const _T* pSrc1, const _T* pSrc2, _T* pDst, int len); // sqtr(a^2+b^2)

        namespace f21
        {
            void inv(const float * pSrc, float * pDst, int len);
            void sqrt(const float * pSrc, float * pDst, int len);
            void invSqrt(const float * pSrc, float * pDst, int len);
            void cbrt(const float * pSrc, float * pDst, int len);
            void powx(const float * pSrc, const float constValue, float * pDst, int len);
            void pow(const float * pSrc1, const float * pSrc2, float * pDst, int len);
            void hypot(const float * pSrc1, const float * pSrc2, float * pDst, int len);
        }
        namespace f24
        {
            void inv(const float * pSrc, float * pDst, int len);
            void sqr(const float * pSrc, float * pDst, int len);
            void sqrt(const float * pSrc, float * pDst, int len);
            void invSqrt(const float * pSrc, float * pDst, int len);
            void cbrt(const float * pSrc, float * pDst, int len);
            void powx(const float * pSrc, const float constValue, float * pDst, int len);
            void pow(const float * pSrc1, const float * pSrc2, float * pDst, int len);
            void hypot(const float * pSrc1, const float * pSrc2, float * pDst, int len);
        }
        namespace d50
        {
            void inv(const double * pSrc, double * pDst, int len);
            void sqrt(const double * pSrc, double * pDst, int len);
            void invSqrt(const double * pSrc, double * pDst, int len);
            void cbrt(const double * pSrc, double * pDst, int len);
            void powx(const double * pSrc, const double constValue, double * pDst, int len);
            void pow(const double * pSrc1, const double * pSrc2, double * pDst, int len);
            void hypot(const double * pSrc1, const double * pSrc2, double * pDst, int len);
        }
        namespace d53
        {
            void inv(const double * pSrc, double * pDst, int len);
            void sqr(const double * pSrc, double * pDst, int len);
            void sqrt(const double * pSrc, double * pDst, int len);
            void invSqrt(const double * pSrc, double * pDst, int len);
            void cbrt(const double * pSrc, double * pDst, int len);
            void powx(const double * pSrc, const double constValue, double * pDst, int len);
            void pow(const double * pSrc1, const double * pSrc2, double * pDst, int len);
            void hypot(const double * pSrc1, const double * pSrc2, double * pDst, int len);
        }
    }

    namespace exp_log
    {
        _SIMD_EXT_T void exp(const _T* pSrc, _T* pDst, int len);
        _SIMD_EXT_T void ln(const _T* pSrc, _T* pDst, int len);

        namespace f21
        {
            void exp(const float * pSrc, float * pDst, int len);
            void ln(const float * pSrc, float * pDst, int len);
        }
        namespace f24
        {
            void exp(const float * pSrc, float * pDst, int len);
            void ln(const float * pSrc, float * pDst, int len);
        }
        namespace d50
        {
            void exp(const double * pSrc, double * pDst, int len);
            void ln(const double * pSrc, double * pDst, int len);
        }
        namespace d53
        {
            void exp(const double * pSrc, double * pDst, int len);
            void ln(const double * pSrc, double * pDst, int len);
        }
    }

    namespace statistical
    {
        _SIMD_EXT_T void max(const _T* pSrc, int len, _T* pMax);
        _SIMD_EXT_T void maxIndx(const _T* pSrc, int len, _T* pMax, int* pIndx);

        _SIMD_EXT_T void min(const _T* pSrc, int len, _T* pMin);
        _SIMD_EXT_T void minIndx(const _T* pSrc, int len, _T* pMin, int* pIndx);

        _SIMD_EXT_T void minMax(const _T* pSrc, int len, _T* pMin, _T* pMax);
        _SIMD_EXT_T void minMaxIndx(const _T* pSrc, int len, _T* pMin, int* pMinIndx, _T* pMax, int* pMaxIndx);

        _SIMD_EXT_TU void sum(const _T* pSrc, int len, _U* pSum);
        _SIMD_EXT_T void mean(const _T* pSrc, int len, _T* pMean);
        _SIMD_EXT_T void stdDev(const _T* pSrc, int len, _T* pStdDev);
        _SIMD_EXT_T void meanStdDev(const _T* pSrc, int len, _T* pMean, _T* pStdDev);

        _SIMD_EXT_TU void normInf(const _T* pSrc, int len, _U* pNorm);
        _SIMD_EXT_TU void normL1(const _T* pSrc, int len, _U* pNorm);
        _SIMD_EXT_TU void normL2(const _T* pSrc, int len, _U* pNorm);

        _SIMD_EXT_TU void normDiffInf(const _T* pSrc1, const _T* pSrc2, int len, _U* pNorm);
        _SIMD_EXT_TU void normDiffL1(const _T* pSrc1, const _T* pSrc2, int len, _U* pNorm);
        _SIMD_EXT_TU void normDiffL2(const _T* pSrc1, const _T* pSrc2, int len, _U* pNorm);
#if 0
        // MahDistSingle (IPPs p.8 Speech Recording > Model Evalution)
#endif

        _SIMD_EXT_TU void dotProd(const _T* pSrc1, const _T* pSrc2, int len, _U* pDp);
    }

    namespace trigonometric
    {
        namespace f21
        {
            void cos(const float * pSrc, float * pDst, int len);
            void sin(const float * pSrc, float * pDst, int len);
            void tan(const float * pSrc, float * pDst, int len);
            void acos(const float * pSrc, float * pDst, int len);
            void asin(const float * pSrc, float * pDst, int len);
            void atan(const float * pSrc, float * pDst, int len);
        }
        namespace f24
        {
            void cos(const float * pSrc, float * pDst, int len);
            void sin(const float * pSrc, float * pDst, int len);
            void tan(const float * pSrc, float * pDst, int len);
            void acos(const float * pSrc, float * pDst, int len);
            void asin(const float * pSrc, float * pDst, int len);
            void atan(const float * pSrc, float * pDst, int len);
        }
        namespace d50
        {
            void cos(const double * pSrc, double * pDst, int len);
            void sin(const double * pSrc, double * pDst, int len);;
            void tan(const double * pSrc, double * pDst, int len);
            void acos(const double * pSrc, double * pDst, int len);
            void asin(const double * pSrc, double * pDst, int len);
            void atan(const double * pSrc, double * pDst, int len);
        }
        namespace d53
        {
            void cos(const double * pSrc, double * pDst, int len);
            void sin(const double * pSrc, double * pDst, int len);
            void tan(const double * pSrc, double * pDst, int len);
            void acos(const double * pSrc, double * pDst, int len);
            void asin(const double * pSrc, double * pDst, int len);
            void atan(const double * pSrc, double * pDst, int len);
        }

        using namespace ipp::trigonometric::f24;
        using namespace ipp::trigonometric::d53;
    }

    using namespace ipp::common;
    using namespace ipp::arithmetic;
    using namespace ipp::power;
    using namespace ipp::exp_log;
    using namespace ipp::statistical;
    using namespace ipp::trigonometric;

    namespace f21
    {
        using namespace ipp::arithmetic::f21;
        using namespace ipp::power::f21;
        using namespace ipp::exp_log::f21;
        using namespace ipp::trigonometric::f21;
    }

    namespace f24
    {
        using namespace ipp::arithmetic::f24;
        using namespace ipp::power::f24;
        using namespace ipp::exp_log::f24;
        using namespace ipp::trigonometric::f24;
    }

    namespace d50
    {
        using namespace ipp::arithmetic::d50;
        using namespace ipp::power::d50;
        using namespace ipp::exp_log::d50;
        using namespace ipp::trigonometric::d50;
    }

    namespace d53
    {
        using namespace ipp::arithmetic::d53;
        using namespace ipp::power::d53;
        using namespace ipp::exp_log::d53;
        using namespace ipp::trigonometric::d53;
    }
}
