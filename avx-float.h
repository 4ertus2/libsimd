#pragma once
#include <immintrin.h>

#include "sse.h"

namespace sse
{
namespace internals
{
    INLINE __m256 abs_ps(__m256 x)
    {
        static const __m256 sign_mask = _mm256_set1_ps(-0.f); // -0.f = 1 << 31
        return _mm256_andnot_ps(sign_mask, x); // !sign_mask & x
    }

    INLINE __m256i tailMask(int len)
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

    template <  IntrAvxS::Unary op>
    INLINE void sValDstTail(const __m256& a, float * pDst, int len)
    {
        __m256i mask = tailMask(len);
        _mm256_maskstore_ps(pDst, mask, op(a));
    }

    template <  IntrAvxS::Unary op>
    INLINE void sPrtDstTail(const float * pSrc, float * pDst, int len)
    {
        __m256i mask = tailMask(len);
        __m256 x = _mm256_maskload_ps(pSrc, mask);
        _mm256_maskstore_ps(pDst, mask, op(x));
    }

    template <  IntrAvxS::Binary op>
    INLINE void sPrtValDstTail(const float * pSrc, const __m256& b, float * pDst, int len)
    {
        __m256i mask = tailMask(len);
        __m256 x = _mm256_maskload_ps(pSrc, mask);
        _mm256_maskstore_ps(pDst, mask, op(x, b));
    }

    template <  IntrAvxS::Binary op>
    INLINE void sPrtPtrDstTail(const float * pSrc1, const float * pSrc2, float * pDst, int len)
    {
        __m256i mask = tailMask(len);
        __m256 x = _mm256_maskload_ps(pSrc1, mask);
        __m256 y = _mm256_maskload_ps(pSrc2, mask);
        _mm256_maskstore_ps(pDst, mask, op(x, y));
    }

    //

    template <  IntrAvxS::Unary op,
                IntrAvxS::Store store = avx_store_ps>
    INLINE void sValDst(float value, float * pDst, int len)
    {
        const int shift = 8;
        int tail = len % shift;
        len /= shift;

        __m256 a = _mm256_set1_ps(value);

        for (; len >= 4; len-=4, pDst+=4*shift)
        {
            store(pDst, op(a));
            store(pDst+1*shift, op(a));
            store(pDst+2*shift, op(a));
            store(pDst+3*shift, op(a));
        }

        if (len >= 2)
        {
            store(pDst, op(a));
            store(pDst+1*shift, op(a));
            len -= 2; pDst += 2*shift;
        }

        if (len)
        {
            store(pDst, op(a));
            pDst += shift;
        }

        if (tail)
            sValDstTail<op>(a, pDst, tail);
        _mm256_zeroall();
    }

    template <  IntrAvxS::Unary op,
                IntrAvxS::Load load = avx_load_ps,
                IntrAvxS::Store store = avx_store_ps>
    INLINE void sPtrDst(const float * pSrc, float * pDst, int len)
    {
        const int shift = 8;
        int tail = len % shift;
        len /= shift;

        for (; len >= 4; len-=4, pSrc+=4*shift, pDst+=4*shift)
        {
            store(pDst, op(load(pSrc)));
            store(pDst+1*shift, op(load(pSrc+1*shift)));
            store(pDst+2*shift, op(load(pSrc+2*shift)));
            store(pDst+3*shift, op(load(pSrc+3*shift)));
        }

        if (len >= 2)
        {
            store(pDst, op(load(pSrc)));
            store(pDst+1*shift, op(load(pSrc+1*shift)));

            len -= 2; pSrc += 2*shift; pDst += 2*shift;
        }

        if (len)
        {
            store(pDst, op(load(pSrc)));
            pSrc += shift; pDst += shift;
        }

        if (tail)
            sPrtDstTail<op>(pSrc, pDst, tail);
        _mm256_zeroall();
    }

    template <  IntrAvxS::Binary op,
                IntrAvxS::Load load = avx_load_ps,
                IntrAvxS::Store store = avx_store_ps>
    INLINE void sPtrValDst(const float * pSrc, float value, float * pDst, int len)
    {
        const int shift = 8;
        int tail = len % shift;
        len /= shift;

        __m256 b = _mm256_set1_ps(value);

        for (; len >= 4; len-=4, pSrc+=4*shift, pDst+=4*shift)
        {
            store(pDst, op(load(pSrc), b));
            store(pDst+1*shift, op(load(pSrc+1*shift), b));
            store(pDst+2*shift, op(load(pSrc+2*shift), b));
            store(pDst+3*shift, op(load(pSrc+3*shift), b));
        }

        if (len >= 2)
        {
            store(pDst, op(load(pSrc), b));
            store(pDst+1*shift, op(load(pSrc+1*shift), b));

            len -= 2; pSrc += 2*shift; pDst += 2*shift;
        }

        if (len)
        {
            store(pDst, op(load(pSrc), b));
            pSrc += shift; pDst += shift;
        }

        if (tail)
            sPrtValDstTail<op>(pSrc, b, pDst, tail);
        _mm256_zeroall();
    }

    template <  IntrAvxS::Binary op,
                IntrAvxS::Load load = avx_load_ps,
                IntrAvxS::Store store = avx_store_ps>
    INLINE void sPtrPtrDst(const float * pSrc1, const float * pSrc2, float * pDst, int len)
    {
        const int shift = 8;
        int tail = len % shift;
        len /= shift;

        for (; len >= 4; len-=4, pSrc1+=4*shift, pSrc2+=4*shift, pDst+=4*shift)
        {
            __m256 a0 = load(pSrc1);
            __m256 a1 = load(pSrc1+1*shift);
            __m256 a2 = load(pSrc1+2*shift);
            __m256 a3 = load(pSrc1+3*shift);

            __m256 b0 = load(pSrc2);
            __m256 b1 = load(pSrc2+1*shift);
            __m256 b2 = load(pSrc2+2*shift);
            __m256 b3 = load(pSrc2+3*shift);

            store(pDst, op(a0, b0));
            store(pDst+1*shift, op(a1, b1));
            store(pDst+2*shift, op(a2, b2));
            store(pDst+3*shift, op(a3, b3));
        }

        if (len >= 2)
        {
            store(pDst, op(load(pSrc1), load(pSrc2)));
            store(pDst+1*shift, op(load(pSrc1+1*shift), load(pSrc2+1*shift)));

            len -= 2; pSrc1 += 2*shift; pSrc2 += 2*shift; pDst += 2*shift;
        }

        if (len)
        {
            store(pDst, op(load(pSrc1), load(pSrc2)));
            pSrc1 += shift; pSrc2 += shift; pDst += shift;
        }

        if (tail)
            sPrtPtrDstTail<op>(pSrc1, pSrc2, pDst, tail);
        _mm256_zeroall();
    }
}

namespace common
{
    _SIMD_SSE_SPEC void set(float val, float * pDst, int len)
    {
        internals::sValDst<nop>(val, pDst, len);
    }

    _SIMD_SSE_SPEC void copy(const float * pSrc, float * pDst, int len)
    {
        internals::sPtrDst<nop>(pSrc, pDst, len);
    }
}

namespace arithmetic
{
    _SIMD_SSE_SPEC void addC(const float * pSrc, float val, float * pDst, int len)
    {
        internals::sPtrValDst<_mm256_add_ps>(pSrc, val, pDst, len);
    }

    _SIMD_SSE_SPEC void subC(const float * pSrc, float val, float * pDst, int len)
    {
        internals::sPtrValDst<_mm256_sub_ps>(pSrc, val, pDst, len);
    }

    _SIMD_SSE_SPEC void mulC(const float * pSrc, float val, float * pDst, int len)
    {
        internals::sPtrValDst<_mm256_mul_ps>(pSrc, val, pDst, len);
    }

    _SIMD_SSE_SPEC void divC(const float * pSrc, float val, float * pDst, int len)
    {
        internals::sPtrValDst<_mm256_div_ps>(pSrc, val, pDst, len);
    }


    _SIMD_SSE_SPEC void subCRev(const float * pSrc, float val, float * pDst, int len)
    {
        internals::sPtrValDst<IntrAvxS::rev_op<_mm256_sub_ps>>(pSrc, val, pDst, len);
    }

    _SIMD_SSE_SPEC void divCRev(const float * pSrc, float val, float * pDst, int len)
    {
        internals::sPtrValDst<IntrAvxS::rev_op<_mm256_div_ps>>(pSrc, val, pDst, len);
    }


    _SIMD_SSE_SPEC void add(const float * pSrc1, const float * pSrc2, float * pDst, int len)
    {
        internals::sPtrPtrDst<_mm256_add_ps>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_SSE_SPEC void sub(const float * pSrc1, const float * pSrc2, float * pDst, int len)
    {
        internals::sPtrPtrDst<_mm256_sub_ps>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_SSE_SPEC void mul(const float * pSrc1, const float * pSrc2, float * pDst, int len)
    {
        internals::sPtrPtrDst<_mm256_mul_ps>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_SSE_SPEC void div(const float * pSrc1, const float * pSrc2, float * pDst, int len)
    {
        internals::sPtrPtrDst<_mm256_div_ps>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_SSE_SPEC void abs(const float * pSrc, float * pDst, int len)
    {
        internals::sPtrDst<internals::abs_ps>(pSrc, pDst, len);
    }
}

namespace power
{
    _SIMD_SSE_SPEC void inv(const float * pSrc, float * pDst, int len)
    {
        internals::sPtrDst<_mm256_rcp_ps>(pSrc, pDst, len);
    }

    _SIMD_SSE_SPEC void sqrt(const float * pSrc, float * pDst, int len)
    {
        internals::sPtrDst<_mm256_sqrt_ps>(pSrc, pDst, len);
    }

    _SIMD_SSE_SPEC void invSqrt(const float * pSrc, float * pDst, int len)
    {
        internals::sPtrDst<_mm256_rsqrt_ps>(pSrc, pDst, len);
    }
}

// TODO
namespace statistical
{
    _SIMD_SSE_SPEC void min(const float * pSrc, int len, float * pMin)
    {
        return nosimd::statistical::min(pSrc, len, pMin);
    }

    _SIMD_SSE_SPEC void max(const float * pSrc, int len, float * pMax)
    {
        return nosimd::statistical::max(pSrc, len, pMax);
    }

    _SIMD_SSE_SPEC void minMax(const float * pSrc, int len, float * pMin, float * pMax)
    {
        return nosimd::statistical::minMax(pSrc, len, pMin, pMax);
    }

    _SIMD_SSE_SPEC void sum(const float * pSrc, int len, float * pSum)
    {
        return nosimd::statistical::sum(pSrc, len, pSum);
    }

    _SIMD_SSE_SPEC void meanStdDev(const float * pSrc, int len, float * pMean, float * pStdDev)
    {
        return nosimd::statistical::meanStdDev(pSrc, len, pMean, pStdDev);
    }

    _SIMD_SSE_SPEC void dotProd(const float * pSrc1, const float * pSrc2, int len, float * pDp)
    {
        return nosimd::statistical::dotProd(pSrc1, pSrc2, len, pDp);
    }
}
}
