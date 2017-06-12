#pragma once
#include <immintrin.h>

#include "sse.h"

namespace sse
{
namespace internals
{
    INLINE __m256d abs_pd(__m256d x)
    {
        static const __m256d sign_mask = _mm256_set1_pd(-0.);
        return _mm256_andnot_pd(sign_mask, x);
    }

    template <  IntrAvxD::Unary op>
    INLINE void dValDstTail(const __m256d& a, double * pDst, int len)
    {
        __m256i mask = avxTailMask64(len);
        _mm256_maskstore_pd(pDst, mask, op(a));
    }

    template <  IntrAvxD::Unary op>
    INLINE void dPrtDstTail(const double * pSrc, double * pDst, int len)
    {
        __m256i mask = avxTailMask64(len);
        __m256d x = _mm256_maskload_pd(pSrc, mask);
        _mm256_maskstore_pd(pDst, mask, op(x));
    }

    template <  IntrAvxD::Binary op>
    INLINE void dPrtValDstTail(const double * pSrc, const __m256d& b, double * pDst, int len)
    {
        __m256i mask = avxTailMask64(len);
        __m256d x = _mm256_maskload_pd(pSrc, mask);
        _mm256_maskstore_pd(pDst, mask, op(x, b));
    }

    template <  IntrAvxD::Binary op>
    INLINE void dPrtPtrDstTail(const double * pSrc1, const double * pSrc2, double * pDst, int len)
    {
        __m256i mask = avxTailMask64(len);
        __m256d x = _mm256_maskload_pd(pSrc1, mask);
        __m256d y = _mm256_maskload_pd(pSrc2, mask);
        _mm256_maskstore_pd(pDst, mask, op(x, y));
    }

    //

    template <  IntrAvxD::Unary op,
                IntrAvxD::Store store = avx_store_pd>
    INLINE void dValDst(double value, double * pDst, int len)
    {
        const int shift = avxBlockLen(double());
        int tail = len % shift;
        len /= shift;

        __m256d a = _mm256_set1_pd(value);

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
            dValDstTail<op>(a, pDst, tail);
        _mm256_zeroall();
    }

    template <  IntrAvxD::Unary op,
                IntrAvxD::Load load = avx_load_pd,
                IntrAvxD::Store store = avx_store_pd>
    INLINE void dPtrDst(const double * pSrc, double * pDst, int len)
    {
        const int shift = avxBlockLen(double());
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
            dPrtDstTail<op>(pSrc, pDst, tail);
        _mm256_zeroall();
    }

    template <  IntrAvxD::Binary op,
                IntrAvxD::Load load = avx_load_pd,
                IntrAvxD::Store store = avx_store_pd>
    INLINE void dPtrValDst(const double * pSrc, double value, double * pDst, int len)
    {
        const int shift = avxBlockLen(double());
        int tail = len % shift;
        len /= shift;

        __m256d b = _mm256_set1_pd(value);

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
            dPrtValDstTail<op>(pSrc, b, pDst, tail);
        _mm256_zeroall();
    }

    template <  IntrAvxD::Binary op,
                IntrAvxD::Load load = avx_load_pd,
                IntrAvxD::Store store = avx_store_pd>
    INLINE void dPtrPtrDst(const double * pSrc1, const double * pSrc2, double * pDst, int len)
    {
        const int shift = avxBlockLen(double());
        int tail = len % shift;
        len /= shift;

        for (; len >= 4; len-=4, pSrc1+=4*shift, pSrc2+=4*shift, pDst+=4*shift)
        {
            __m256d a0 = load(pSrc1);
            __m256d a1 = load(pSrc1+1*shift);
            __m256d a2 = load(pSrc1+2*shift);
            __m256d a3 = load(pSrc1+3*shift);

            __m256d b0 = load(pSrc2);
            __m256d b1 = load(pSrc2+1*shift);
            __m256d b2 = load(pSrc2+2*shift);
            __m256d b3 = load(pSrc2+3*shift);

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
            dPrtPtrDstTail<op>(pSrc1, pSrc2, pDst, tail);
        _mm256_zeroall();
    }
}

namespace common
{
    _SIMD_SSE_SPEC void set(double val, double * pDst, int len)
    {
        internals::dValDst<nop>(val, pDst, len);
    }

    _SIMD_SSE_SPEC void copy(const double * pSrc, double * pDst, int len)
    {
        internals::dPtrDst<nop>(pSrc, pDst, len);
    }
}

namespace arithmetic
{
    _SIMD_SSE_SPEC void addC(const double * pSrc, double val, double * pDst, int len)
    {
        internals::dPtrValDst<_mm256_add_pd>(pSrc, val, pDst, len);
    }

    _SIMD_SSE_SPEC void subC(const double * pSrc, double val, double * pDst, int len)
    {
        internals::dPtrValDst<_mm256_sub_pd>(pSrc, val, pDst, len);
    }

    _SIMD_SSE_SPEC void mulC(const double * pSrc, double val, double * pDst, int len)
    {
        internals::dPtrValDst<_mm256_mul_pd>(pSrc, val, pDst, len);
    }

    _SIMD_SSE_SPEC void divC(const double * pSrc, double val, double * pDst, int len)
    {
        internals::dPtrValDst<_mm256_div_pd>(pSrc, val, pDst, len);
    }


    _SIMD_SSE_SPEC void subCRev(const double * pSrc, double val, double * pDst, int len)
    {
        internals::dPtrValDst<IntrAvxD::rev_op<_mm256_sub_pd>>(pSrc, val, pDst, len);
    }

    _SIMD_SSE_SPEC void divCRev(const double * pSrc, double val, double * pDst, int len)
    {
        internals::dPtrValDst<IntrAvxD::rev_op<_mm256_div_pd>>(pSrc, val, pDst, len);
    }


    _SIMD_SSE_SPEC void add(const double * pSrc1, const double * pSrc2, double * pDst, int len)
    {
        internals::dPtrPtrDst<_mm256_add_pd>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_SSE_SPEC void sub(const double * pSrc1, const double * pSrc2, double * pDst, int len)
    {
        internals::dPtrPtrDst<_mm256_sub_pd>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_SSE_SPEC void mul(const double * pSrc1, const double * pSrc2, double * pDst, int len)
    {
        internals::dPtrPtrDst<_mm256_mul_pd>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_SSE_SPEC void div(const double * pSrc1, const double * pSrc2, double * pDst, int len)
    {
        internals::dPtrPtrDst<_mm256_div_pd>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_SSE_SPEC void abs(const double * pSrc, double * pDst, int len)
    {
        internals::dPtrDst<internals::abs_pd>(pSrc, pDst, len);
    }
}

namespace power
{
    _SIMD_SSE_SPEC void inv(const double * pSrc, double * pDst, int len)
    {
        divCRev(pSrc, 1.0, pDst, len);
    }

    _SIMD_SSE_SPEC void sqrt(const double * pSrc, double * pDst, int len)
    {
        internals::dPtrDst<_mm256_sqrt_pd>(pSrc, pDst, len);
    }

    _SIMD_SSE_SPEC void invSqrt(const double * pSrc, double * pDst, int len)
    {
        sqrt(pSrc, pDst, len);
        inv(pDst, pDst, len);
    }
}

namespace statistical
{
    _SIMD_SSE_SPEC void min(const double * pSrc, int len, double * pMin)
    {
        return nosimd::statistical::min(pSrc, len, pMin);
    }

    _SIMD_SSE_SPEC void max(const double * pSrc, int len, double * pMax)
    {
        return nosimd::statistical::max(pSrc, len, pMax);
    }

    _SIMD_SSE_SPEC void minMax(const double * pSrc, int len, double * pMin, double * pMax)
    {
        return nosimd::statistical::minMax(pSrc, len, pMin, pMax);
    }

    _SIMD_SSE_SPEC void sum(const double * pSrc, int len, double * pSum)
    {
        return nosimd::statistical::sum(pSrc, len, pSum);
    }

    _SIMD_SSE_SPEC void meanStdDev(const double * pSrc, int len, double * pMean, double * pStdDev)
    {
        return nosimd::statistical::meanStdDev(pSrc, len, pMean, pStdDev);
    }

    _SIMD_SSE_SPEC void dotProd(const double * pSrc1, const double * pSrc2, int len, double * pDp)
    {
        return nosimd::statistical::dotProd(pSrc1, pSrc2, len, pDp);
    }
}
}
