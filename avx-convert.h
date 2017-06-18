#pragma once
#include <immintrin.h>

#include "sse.h"

namespace sse
{
namespace internals
{
    INLINE void convertTail(const float * pSrc, double * pDst, int len)
    {
        __m128 a = _mm_maskload_ps(pSrc, sseTailMask32(len));
        __m256d b = _mm256_cvtps_pd(a);
        _mm256_maskstore_pd(pDst, avxTailMask64(len), b);
    }

    INLINE void convertTail(const float * pSrc, int32_t * pDst, int len)
    {
        __m256i mask = avxTailMask32(len);
        __m256 a = _mm256_maskload_ps(pSrc, mask);
        __m256i b = _mm256_cvtps_epi32(a);
        _mm256_maskstore_epi32(pDst, mask, b);
    }

    INLINE void convertTail(const int32_t * pSrc, float * pDst, int len)
    {
        __m256i mask = avxTailMask32(len);
        __m256i a = _mm256_maskload_epi32(pSrc, mask);
        __m256 b = _mm256_cvtepi32_ps(a);
        _mm256_maskstore_ps(pDst, mask, b);
    }

    INLINE void convertTail(const int32_t * pSrc, double * pDst, int len)
    {
        __m128i a = _mm_maskload_epi32(pSrc, sseTailMask32(len));
        __m256d b = _mm256_cvtepi32_pd(a);
        _mm256_maskstore_pd(pDst, avxTailMask64(len), b);
    }

    INLINE void convertTail(const double * pSrc, float * pDst, int len)
    {
        __m256d a = _mm256_maskload_pd(pSrc, avxTailMask64(len));
        __m128 b = _mm256_cvtpd_ps(a);
        _mm_maskstore_ps(pDst, sseTailMask32(len), b);
    }

    INLINE void convertTail(const double * pSrc, int32_t * pDst, int len)
    {
        __m256d a = _mm256_maskload_pd(pSrc, avxTailMask64(len));
        __m128i b = _mm256_cvtpd_epi32(a);
        _mm_maskstore_epi32(pDst, sseTailMask32(len), b);
    }

    //

    INLINE void convert(const float * pSrc, double * pDst, int len)
    {
        int tail = len % 4;
        const float * pEnd = pSrc + (len-tail);

        for (; pSrc < pEnd; pSrc += 4, pDst += 4)
        {
            __m128 a = sse_load_ps(pSrc);
            __m256d b = _mm256_cvtps_pd(a);
            avx_store_pd(pDst, b);
        }

        convertTail(pSrc, pDst, tail);
    }

    INLINE void convert(const float * pSrc, int32_t * pDst, int len)
    {
        int tail = len % 8;
        const float * pEnd = pSrc + (len-tail);

        for (; pSrc < pEnd; pSrc += 8, pDst += 8)
        {
            __m256 a = avx_load_ps(pSrc);
            __m256i b = _mm256_cvtps_epi32(a);
            avx_store_si((__m256i*)pDst, b);
        }

        convertTail(pSrc, pDst, tail);
    }

    INLINE void convert(const int32_t * pSrc, float * pDst, int len)
    {
        int tail = len % 8;
        const int32_t * pEnd = pSrc + (len-tail);

        for (; pSrc < pEnd; pSrc += 8, pDst += 8)
        {
            __m256i a = avx_load_si((__m256i*)pSrc);
            __m256 b = _mm256_cvtepi32_ps(a);
            avx_store_ps(pDst, b);
        }

        convertTail(pSrc, pDst, tail);
    }

    INLINE void convert(const int32_t * pSrc, double * pDst, int len)
    {
        int tail = len % 4;
        const int32_t * pEnd = pSrc + (len-tail);

        for (; pSrc < pEnd; pSrc += 4, pDst += 4)
        {
            __m128i a = sse_load_si((const __m128i *)pSrc);
            __m256d b = _mm256_cvtepi32_pd(a);
            avx_store_pd(pDst, b);
        }

        convertTail(pSrc, pDst, tail);
    }

    INLINE void convert(const double * pSrc, float * pDst, int len)
    {
        int tail = len % 4;
        const double * pEnd = pSrc + (len-tail);

        for (; pSrc < pEnd; pSrc += 4, pDst += 4) {
            __m256d a = avx_load_pd(pSrc);
            __m128 b = _mm256_cvtpd_ps(a);
            sse_store_ps(pDst, b);
        }

        convertTail(pSrc, pDst, tail);
    }

    INLINE void convert(const double * pSrc, int32_t * pDst, int len)
    {
        int tail = len % 4;
        const double * pEnd = pSrc + (len-tail);

        for (; pSrc < pEnd; pSrc += 4, pDst += 4) {
            __m256d a = avx_load_pd(pSrc);
            __m128i b = _mm256_cvtpd_epi32(a);
            sse_store_si((__m128i*)pDst, b);
        }

        convertTail(pSrc, pDst, tail);
    }

#if 0
    INLINE void convert_v2(const float * pSrc, double * pDst, int len)
    {
        int tail = len % 8;
        const float * pEnd = pSrc + (len-tail);

        for (; pSrc < pEnd; pSrc += 8)
        {
            __m256 a = avx_load_ps(pSrc);
            __m128 a0 = _mm256_castps256_ps128(a);
            __m128 a1 = _mm256_extractf128_ps(a, 1);
            __m256d b0 = _mm256_cvtps_pd(a0);
            __m256d b1 = _mm256_cvtps_pd(a1);

            avx_store_pd(pDst, b0);
            avx_store_pd(pDst + 4, b1);
            pDst += 8;
        }

        if (tail > 4)
        {
            __m128 a0 = sse_load_ps(pSrc);
            __m256d b0 = _mm256_cvtps_pd(a0);
            avx_store_pd(pDst, b0);

            tail -= 4; pSrc += 4; pDst += 4;
        }

        convertTail(pSrc, pDst, tail);
    }

    INLINE void convert_v2(const double * pSrc, float * pDst, int len)
    {
        int tail = len % 8;
        const double * pEnd = pSrc + (len-tail);

        for (; pSrc < pEnd; pSrc += 8) {
            __m256d a0 = avx_load_pd(pSrc);
            __m256d a1 = avx_load_pd(pSrc+4);

            __m128 b0 = _mm256_cvtpd_ps(a0);
            __m128 b1 = _mm256_cvtpd_ps(a1);
            __m256 b = _mm256_insertf128_ps(_mm256_castps128_ps256(b0), b1, 1);

            avx_store_ps(pDst, b);
            pDst += 8;
        }

        if (tail > 4)
        {
            __m256d a = avx_load_pd(pSrc);
            __m128 b = _mm256_cvtpd_ps(a);
            sse_store_ps(pDst, b);

            tail -= 4; pSrc += 4; pDst += 4;
        }

        convertTail(pSrc, pDst, tail);
    }
#endif
}

namespace common
{
    _SIMD_SSE_SPEC void convert(const float * pSrc, double * pDst, int len)
    {
#if 1
        internals::convert(pSrc, pDst, len);
#else
        internals::convert_v2(pSrc, pDst, len);
#endif
    }

    _SIMD_SSE_SPEC void convert(const float * pSrc, int32_t * pDst, int len)
    {
        internals::convert(pSrc, pDst, len);
    }

    _SIMD_SSE_SPEC void convert(const double * pSrc, float * pDst, int len)
    {
#if 1
        internals::convert(pSrc, pDst, len);
#else
        internals::convert_v2(pSrc, pDst, len);
#endif
    }

    _SIMD_SSE_SPEC void convert(const double * pSrc, int32_t * pDst, int len)
    {
        internals::convert(pSrc, pDst, len);
    }

    _SIMD_SSE_SPEC void convert(const int32_t * pSrc, float * pDst, int len)
    {
        internals::convert(pSrc, pDst, len);
    }

    _SIMD_SSE_SPEC void convert(const int32_t * pSrc, double * pDst, int len)
    {
        internals::convert(pSrc, pDst, len);
    }

    //

    // TODO: 16 <-> float, 8 <-> float (convert + convert)
    // TODO: 32 -> 16, 16 -> 8 _mm256_packs_epi*

    //

    _SIMD_SSE_SPEC void convert(const int8_t * pSrc, int16_t * pDst, int len)
    {
        nosimd::common::convert(pSrc, pDst, len);
    }

    _SIMD_SSE_SPEC void convert(const int8_t * pSrc, int32_t * pDst, int len)
    {
        nosimd::common::convert(pSrc, pDst, len);
    }

    _SIMD_SSE_SPEC void convert(const int8_t * pSrc, int64_t * pDst, int len)
    {
        nosimd::common::convert(pSrc, pDst, len);
    }

    _SIMD_SSE_SPEC void convert(const uint8_t * pSrc, uint16_t * pDst, int len)
    {
        nosimd::common::convert(pSrc, pDst, len);
    }

    _SIMD_SSE_SPEC void convert(const uint8_t * pSrc, uint32_t * pDst, int len)
    {
        nosimd::common::convert(pSrc, pDst, len);
    }

    _SIMD_SSE_SPEC void convert(const uint8_t * pSrc, uint64_t * pDst, int len)
    {
        nosimd::common::convert(pSrc, pDst, len);
    }

    _SIMD_SSE_SPEC void convert(const int16_t * pSrc, int8_t * pDst, int len)
    {
        nosimd::common::convert(pSrc, pDst, len);
    }

    _SIMD_SSE_SPEC void convert(const int16_t * pSrc, int32_t * pDst, int len)
    {
        nosimd::common::convert(pSrc, pDst, len);
    }

    _SIMD_SSE_SPEC void convert(const int16_t * pSrc, int64_t * pDst, int len)
    {
        nosimd::common::convert(pSrc, pDst, len);
    }

    _SIMD_SSE_SPEC void convert(const uint16_t * pSrc, uint8_t * pDst, int len)
    {
        nosimd::common::convert(pSrc, pDst, len);
    }

    _SIMD_SSE_SPEC void convert(const uint16_t * pSrc, uint32_t * pDst, int len)
    {
        nosimd::common::convert(pSrc, pDst, len);
    }

    _SIMD_SSE_SPEC void convert(const uint16_t * pSrc, uint64_t * pDst, int len)
    {
        nosimd::common::convert(pSrc, pDst, len);
    }

    _SIMD_SSE_SPEC void convert(const int32_t * pSrc, int16_t * pDst, int len)
    {
        nosimd::common::convert(pSrc, pDst, len);
    }

    _SIMD_SSE_SPEC void convert(const int32_t * pSrc, int64_t * pDst, int len)
    {
        nosimd::common::convert(pSrc, pDst, len);
    }

    _SIMD_SSE_SPEC void convert(const uint32_t * pSrc, uint16_t * pDst, int len)
    {
        nosimd::common::convert(pSrc, pDst, len);
    }

    _SIMD_SSE_SPEC void convert(const uint32_t * pSrc, uint64_t * pDst, int len)
    {
        nosimd::common::convert(pSrc, pDst, len);
    }
}
}
