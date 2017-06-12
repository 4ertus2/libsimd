#pragma once
#include "nosimd.h"

#ifdef max
#undef max
#endif

#ifdef min
#undef min
#endif

#if __cplusplus >= 201103L
#define INLINE __attribute__((always_inline)) inline
#else
#define INLINE inline
#endif

#ifndef _SIMD_SSE_T
#define _SIMD_SSE_T template<typename _T> inline
#define _SIMD_SSE_TU template<typename _T, typename _U> inline
#define _SIMD_SSE_SPEC template <> INLINE
#endif

namespace sse
{
    template <typename _T>
    INLINE _T nop(_T x) { return x; }

    template <typename _T, typename _U>
    struct Intrinsic
    {
        typedef _T (*Unary)(_T);
        typedef _T (*Binary)(_T, _T);
        typedef _T (*Load)(const _U *);
        typedef void (*Store)(_U *, _T);

        template<Binary op>
        static INLINE _T rev_op(_T x, _T y) { return op(y, x); }
    };

    using IntrS = Intrinsic<__m128, float>;
    using IntrD = Intrinsic<__m128d, double>;
    using IntrI = Intrinsic<__m128i, __m128i>;

#ifdef SSE_ALIGNED
    INLINE __m128 sse_load_ps(float const * x) { return _mm_load_ps(x); }
    INLINE __m128d sse_load_pd(double const * x) { return _mm_load_pd(x); }
    INLINE __m128i sse_load_si(const __m128i * x) { return _mm_load_si128(x); }
    INLINE void sse_store_ps(float * x, __m128 y) { _mm_store_ps(x, y); }
    INLINE void sse_store_pd(double * x, __m128d y) { _mm_store_pd(x, y); }
    INLINE void sse_store_si(__m128i * x, __m128i y) { _mm_store_si128(x, y); }
#else
    INLINE __m128 sse_load_ps(float const * x) { return _mm_loadu_ps(x); }
    INLINE __m128d sse_load_pd(double const * x) { return _mm_loadu_pd(x); }
    INLINE __m128i sse_load_si(const __m128i * x) { return _mm_loadu_si128(x); }
    INLINE void sse_store_ps(float * x, __m128 y) { _mm_storeu_ps(x, y); }
    INLINE void sse_store_pd(double * x, __m128d y) { _mm_storeu_pd(x, y); }
    INLINE void sse_store_si(__m128i * x, __m128i y) { _mm_storeu_si128(x, y); }
#endif

#ifdef SIMD_AVX
    using IntrAvxS = Intrinsic<__m256, float>;
    using IntrAvxD = Intrinsic<__m256d, double>;
    using IntrAvxI = Intrinsic<__m256i, __m256i>;

#ifdef SSE_ALIGNED
    INLINE __m256 avx_load_ps(float const * x) { return _mm256_load_ps(x); }
    INLINE __m256d avx_load_pd(double const * x) { return _mm256_load_pd(x); }
    INLINE __m256i avx_load_si(const __m256i * x) { return _mm256_load_si256(x); }
    INLINE void avx_store_ps(float * x, __m256 y) { _mm256_store_ps(x, y); }
    INLINE void avx_store_pd(double * x, __m256d y) { _mm256_store_pd(x, y); }
    INLINE void avx_store_si(__m256i * x, __m256i y) { _mm256_store_si256(x, y); }
#else
    INLINE __m256 avx_load_ps(float const * x) { return _mm256_loadu_ps(x); }
    INLINE __m256d avx_load_pd(double const * x) { return _mm256_loadu_pd(x); }
    INLINE __m256i avx_load_si(const __m256i * x) { return _mm256_loadu_si256(x); }
    INLINE void avx_store_ps(float * x, __m256 y) { _mm256_storeu_ps(x, y); }
    INLINE void avx_store_pd(double * x, __m256d y) { _mm256_storeu_pd(x, y); }
    INLINE void avx_store_si(__m256i * x, __m256i y) { _mm256_storeu_si256(x, y); }
#endif

    INLINE constexpr int avxBlockLen(int8_t) { return 32; }
    INLINE constexpr int avxBlockLen(uint8_t) { return 32; }
    INLINE constexpr int avxBlockLen(int16_t) { return 16; }
    INLINE constexpr int avxBlockLen(uint16_t) { return 16; }
    INLINE constexpr int avxBlockLen(float) { return 8; }
    INLINE constexpr int avxBlockLen(int32_t) { return 8; }
    INLINE constexpr int avxBlockLen(uint32_t) { return 8; }
    INLINE constexpr int avxBlockLen(double) { return 4; }
    INLINE constexpr int avxBlockLen(int64_t) { return 4; }
    INLINE constexpr int avxBlockLen(uint64_t) { return 4; }

    INLINE __m256i avxTailMask32(int len)
    {
        switch (len) {
            case 1:
                return _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, -1);
            case 2:
                return _mm256_set_epi32(0, 0, 0, 0, 0, 0, -1, -1);
            case 3:
                return _mm256_set_epi32(0, 0, 0, 0, 0, -1, -1, -1);
            case 4:
                return _mm256_set_epi32(0, 0, 0, 0, -1, -1, -1, -1);
            case 5:
                return _mm256_set_epi32(0, 0, 0, -1, -1, -1, -1, -1);
            case 6:
                return _mm256_set_epi32(0, 0, -1, -1, -1, -1, -1, -1);
            case 7:
                return _mm256_set_epi32(0, -1, -1, -1, -1, -1, -1, -1);
        }
        return _mm256_setzero_si256();
    }

    INLINE __m256i avxTailMask64(int len)
    {
        switch (len) {
            case 1:
                return _mm256_set_epi64x(0, 0, 0, -1);
            case 2:
                return _mm256_set_epi64x(0, 0, -1, -1);
            case 3:
                return _mm256_set_epi64x(0, -1, -1, -1);
        }
        return _mm256_setzero_si256();
    }

#endif // SIMD_AVX

    //

    namespace common
    {
        _SIMD_SSE_T void set(_T val, _T* pDst, int len);
        _SIMD_SSE_T void copy(const _T* pSrc, _T* pDst, int len);

        _SIMD_SSE_T void zero(_T* pDst, int len)
        {
            set<_T>((_T)0, pDst, len);
        }
#if SSE_ALIGNED
        _SIMD_SSE_T _T* malloc(int len)
        {
            return (_T*)_mm_malloc(len * sizeof(_T), SSE_ALIGNED);
        }

        _SIMD_SSE_T void free(_T* ptr)
        {
            _mm_free(ptr);
        }
#else
        using nosimd::common::malloc;
        using nosimd::common::free;
#endif

#if 0 // TODO
        _SIMD_SSE_T void move(const _T * pSrc, _T * pDst, int len);
        _SIMD_SSE_TU void convert(const _T * pSrc, _U * pDst, int len);
#else
        using nosimd::common::move;
        using nosimd::common::convert;
#endif
    }

    namespace compare
    {
#if 0 // TODO
        _SIMD_SSE_T void find(_T * pSrc, _T val, int len, int * pPosition);
        _SIMD_SSE_T void findNot(_T * pSrc, _T val, int len, int * pPosition);
        _SIMD_SSE_T void findSame(const _T * pSrc1, _T * pSrc2, int len, int * pPosition);
        _SIMD_SSE_T void findDiff(const _T * pSrc1, _T * pSrc2, int len, int * pPosition);
#else
        using nosimd::compare::find;
        using nosimd::compare::findNot;
        using nosimd::compare::findSame;
        using nosimd::compare::findDiff;
#endif
    }

    namespace arithmetic
    {
        _SIMD_SSE_T void addC(const _T* pSrc, _T val, _T* pDst, int len);
        _SIMD_SSE_T void add(const _T* pSrc1, const _T* pSrc2, _T* pDst, int len);

        _SIMD_SSE_T void subC(const _T* pSrc, _T val, _T* pDst, int len);
        _SIMD_SSE_T void subCRev(const _T* pSrc, _T val, _T* pDst, int len);
        _SIMD_SSE_T void sub(const _T* pSrc1, const _T* pSrc2, _T* pDst, int len);

        _SIMD_SSE_T void mulC(const _T* pSrc, _T val, _T* pDst, int len);
        _SIMD_SSE_T void mul(const _T* pSrc1, const _T* pSrc2, _T* pDst, int len);

        _SIMD_SSE_T void divC(const _T* pSrc, _T val, _T* pDst, int len);
        _SIMD_SSE_T void divCRev(const _T* pSrc, _T val, _T* pDst, int len);
        _SIMD_SSE_T void div(const _T* pSrc1, const _T* pSrc2, _T* pDst, int len);

        _SIMD_SSE_T void abs(const _T* pSrc, _T* pDst, int len);
    }

    namespace power
    {
        _SIMD_SSE_T void inv(const _T* pSrc, _T* pDst, int len);
        _SIMD_SSE_T void sqrt(const _T* pSrc, _T* pDst, int len);
        _SIMD_SSE_T void invSqrt(const _T* pSrc, _T* pDst, int len);

        using nosimd::power::powx;
        using nosimd::power::pow;
        using nosimd::power::cbrt;
        using nosimd::power::hypot;
    }

    namespace statistical
    {
        _SIMD_SSE_T void min(const _T* pSrc, int len, _T* pMin);
        _SIMD_SSE_T void max(const _T* pSrc, int len, _T* pMax);
        _SIMD_SSE_T void minMax(const _T* pSrc, int len, _T* pMin, _T* pMax);

#if 0 // TODO
        _SIMD_SSE_T void minIndx(const _T * pSrc, int len, _T * pMin, int * pIndx);
        _SIMD_SSE_T void maxIndx(const _T * pSrc, int len, _T * pMax, int * pIndx);
        _SIMD_SSE_T void minMaxIndx(const _T * pSrc, int len, _T * pMin, int * pMinIndx, _T * pMax, int * pMaxIndx);
#else
        using nosimd::statistical::minIndx;
        using nosimd::statistical::maxIndx;
        using nosimd::statistical::minMaxIndx;
#endif

        _SIMD_SSE_T void sum(const _T* pSrc, int len, _T* pSum);
        _SIMD_SSE_T void meanStdDev(const _T* pSrc, int len, _T* pMean, _T* pStdDev);

        template<typename _T> inline void mean(const _T* pSrc, int len, _T* pMean)
        {
            sum(pSrc, len, pMean);
            *pMean /= len;
        }

        template<typename _T> inline void stdDev(const _T* pSrc, int len, _T* pStdDev)
        {
            _T m;
            meanStdDev(pSrc, len, &m, pStdDev);
        }

        _SIMD_SSE_T void dotProd(const _T* pSrc1, const _T* pSrc2, int len, _T* pDp);

#if 0 // TODO
        _SIMD_SSE_T void normInf(const _T* pSrc, int len, _T* pNorm);
        _SIMD_SSE_T void normL1(const _T* pSrc, int len, _T* pNorm);
        _SIMD_SSE_T void normL2(const _T* pSrc, int len, _T* pNorm);
        _SIMD_SSE_T void normDiffInf(const _T* pSrc1, const _T* pSrc2, int len, _T* pNorm);
        _SIMD_SSE_T void normDiffL1(const _T* pSrc1, const _T* pSrc2, int len, _T* pNorm);
        _SIMD_SSE_T void normDiffL2(const _T* pSrc1, const _T* pSrc2, int len, _T* pNorm);
#else
        using nosimd::statistical::normInf;
        using nosimd::statistical::normL1;
        using nosimd::statistical::normL2;
        using nosimd::statistical::normDiffInf;
        using nosimd::statistical::normDiffL1;
        using nosimd::statistical::normDiffL2;
#endif
    }

    using namespace sse::common;
    using namespace sse::compare;
    using namespace sse::arithmetic;
    using namespace sse::power;
    using namespace sse::statistical;

    using namespace nosimd::exp_log;
    using namespace nosimd::trigonometric;
}
