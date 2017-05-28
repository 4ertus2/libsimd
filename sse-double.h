#pragma once
#include <cstdint>
#include <emmintrin.h>

#include "sse.h"

namespace sse
{
namespace internals
{
	INLINE  __m128d abs_pd(__m128d x)
	{
		static const __m128d sign_mask = _mm_set1_pd(-0.); // -0. = 1 << 63
		return _mm_andnot_pd(sign_mask, x);
	}

	//

	template <	IntrD::Unary op_pd,
				IntrD::Unary op_sd,
				IntrD::Load load_pd = sse_load_pd,
				IntrD::Store store_pd = sse_store_pd>
	INLINE void dPtrDst(const double * pSrc, double * pDst, int len)
	{
#ifdef UNROLL_MORE
		for (; len >= 8; len-=8, pSrc+=8, pDst+=8)
		{
			__m128d a0 = load_pd(pSrc);
			__m128d a1 = load_pd(pSrc+2);
			__m128d a2 = load_pd(pSrc+4);
			__m128d a3 = load_pd(pSrc+6);

			a0 = abs_pd(a0);
			a1 = abs_pd(a1);
			a2 = abs_pd(a2);
			a3 = abs_pd(a3);

			store_pd(pDst, a0);
			store_pd(pDst+2, a1);
			store_pd(pDst+4, a2);
			store_pd(pDst+6, a3);
		}
#endif
		for (; len >= 4; len-=4, pSrc+=4, pDst+=4)
		{
			__m128d a0 = load_pd(pSrc);
			__m128d a1 = load_pd(pSrc+2);

			a0 = abs_pd(a0);
			a1 = abs_pd(a1);

			store_pd(pDst, a0);
			store_pd(pDst+2, a1);
		}

		if (len >= 2)
		{
			__m128d a0 = load_pd(pSrc);
			a0 = abs_pd(a0);
			store_pd(pDst, a0);

			len -= 2; pSrc += 2; pDst += 2;
		}

		if (len)
		{
			__m128d a0 = _mm_load_sd(pSrc);
			a0 = abs_pd(a0);
			_mm_store_sd(pDst, a0);
		}
	}

	template <	IntrD::Binary op_pd,
				IntrD::Binary op_sd,
				IntrD::Load load_pd = sse_load_pd,
				IntrD::Store store_pd = sse_store_pd>
	INLINE void dPtrValDst(const double * pSrc, double val, double * pDst, int len)
	{
		const __m128d b = _mm_set1_pd(val);
#ifdef UNROLL_MORE
		for (; len >= 8; len-=8, pSrc+=8, pDst+=8)
		{
			__m128d a0 = load_pd(pSrc);
			__m128d a1 = load_pd(pSrc+2);
			__m128d a2 = load_pd(pSrc+4);
			__m128d a3 = load_pd(pSrc+6);

			a0 = op_pd(a0, b);
			a1 = op_pd(a1, b);
			a2 = op_pd(a2, b);
			a3 = op_pd(a3, b);

			store_pd(pDst, a0);
			store_pd(pDst+2, a1);
			store_pd(pDst+4, a2);
			store_pd(pDst+6, a3);
		}
#endif
		for (; len >= 4; len-=4, pSrc+=4, pDst+=4)
		{
			__m128d a0 = load_pd(pSrc);
			__m128d a1 = load_pd(pSrc+2);

			a0 = op_pd(a0, b);
			a1 = op_pd(a1, b);

			store_pd(pDst, a0);
			store_pd(pDst+2, a1);
		}

		if (len >= 2)
		{
			__m128d a0 = load_pd(pSrc);
			a0 = op_pd(a0, b);
			store_pd(pDst, a0);

			len -= 2; pSrc += 2; pDst += 2;
		}

		if (len)
		{
			__m128d a0 = _mm_load_sd(pSrc);
			a0 = op_sd(a0, b);
			_mm_store_sd(pDst, a0);
		}
	}

	template <	IntrD::Binary op_pd,
				IntrD::Binary op_sd,
				IntrD::Load load_pd = sse_load_pd,
				IntrD::Store store_pd = sse_store_pd>
	INLINE void dPtrPtrDst(const double * pSrc1, const double * pSrc2, double * pDst, int len)
	{
#ifdef UNROLL_MORE
		for (; len >= 8; len-=8, pSrc1+=8, pSrc2+=8, pDst+=8)
		{
			__m128d a0 = load_pd(pSrc1);
			__m128d a1 = load_pd(pSrc1+2);
			__m128d a2 = load_pd(pSrc1+4);
			__m128d a3 = load_pd(pSrc1+6);

			__m128d b0 = load_pd(pSrc2);
			__m128d b1 = load_pd(pSrc2+2);
			__m128d b2 = load_pd(pSrc2+4);
			__m128d b3 = load_pd(pSrc2+6);

			a0 = op_pd(a0, b0);
			a1 = op_pd(a1, b1);
			a2 = op_pd(a2, b2);
			a3 = op_pd(a3, b3);

			store_pd(pDst, a0);
			store_pd(pDst+2, a1);
			store_pd(pDst+4, a2);
			store_pd(pDst+6, a3);
		}
#endif
		for (; len >= 4; len-=4, pSrc1+=4, pSrc2+=4, pDst+=4)
		{
			__m128d a0 = load_pd(pSrc1);
			__m128d a1 = load_pd(pSrc1+2);

			__m128d b0 = load_pd(pSrc2);
			__m128d b1 = load_pd(pSrc2+2);

			a0 = op_pd(a0, b0);
			a1 = op_pd(a1, b1);

			store_pd(pDst, a0);
			store_pd(pDst+2, a1);
		}

		if (len >= 2)
		{
			__m128d a0 = load_pd(pSrc1);
			__m128d b0 = load_pd(pSrc2);

			a0 = op_pd(a0, b0);
			store_pd(pDst, a0);

			len -= 2; pSrc1 += 2; pSrc2 += 2; pDst += 2;
 		}

		if (len)
		{
			__m128d a0 = _mm_load_sd(pSrc1);
			__m128d b0 = _mm_load_sd(pSrc2);

			a0 = op_sd(a0, b0);
			_mm_store_sd(pDst, a0);
		}
	}
}

namespace common
{
	template <IntrD::Store store_pd = sse_store_pd>
	INLINE void setT(double val, double * pDst, int len)
	{
		__m128d a = _mm_set1_pd(val);

		for (; len >= 8; len-=8, pDst+=8)
		{
			store_pd(pDst, a);
			store_pd(pDst+2, a);
			store_pd(pDst+4, a);
			store_pd(pDst+6, a);
		}

		if (len >= 4)
		{
			store_pd(pDst, a);
			store_pd(pDst+2, a);
			len -= 4; pDst += 4;
		}

		if (len >= 2)
		{
			store_pd(pDst, a);
			len -= 2; pDst += 2;
		}

		if (len)
			*pDst = val;
	}

	_SIMD_SSE_SPEC void set(double val, double * pDst, int len)
	{
		setT(val, pDst, len);
	}

	_SIMD_SSE_SPEC void copy(const double * pSrc, double * pDst, int len)
	{
		internals::dPtrDst<nop, nop>(pSrc, pDst, len);
	}
}

namespace arithmetic
{
	_SIMD_SSE_SPEC void addC(const double * pSrc, double val, double * pDst, int len)
	{
		internals::dPtrValDst<_mm_add_pd, _mm_add_sd>(pSrc, val, pDst, len);
	}

	_SIMD_SSE_SPEC void subC(const double * pSrc, double val, double * pDst, int len)
	{
		internals::dPtrValDst<_mm_sub_pd, _mm_sub_sd>(pSrc, val, pDst, len);
	}

	_SIMD_SSE_SPEC void mulC(const double * pSrc, double val, double * pDst, int len)
	{
		internals::dPtrValDst<_mm_mul_pd, _mm_mul_sd>(pSrc, val, pDst, len);
	}

	_SIMD_SSE_SPEC void divC(const double * pSrc, double val, double * pDst, int len)
	{
		internals::dPtrValDst<_mm_div_pd, _mm_div_sd>(pSrc, val, pDst, len);
	}


	_SIMD_SSE_SPEC void subCRev(const double * pSrc, double val, double * pDst, int len)
	{
		internals::dPtrValDst<IntrD::rev_op<_mm_sub_pd>, IntrD::rev_op<_mm_sub_sd>>(pSrc, val, pDst, len);
	}

	_SIMD_SSE_SPEC void divCRev(const double * pSrc, double val, double * pDst, int len)
	{
		internals::dPtrValDst<IntrD::rev_op<_mm_div_pd>, IntrD::rev_op<_mm_div_sd>>(pSrc, val, pDst, len);
	}


	_SIMD_SSE_SPEC void add(const double * pSrc1, const double * pSrc2, double * pDst, int len)
	{
		internals::dPtrPtrDst<_mm_add_pd, _mm_add_sd>(pSrc1, pSrc2, pDst, len);
	}

	_SIMD_SSE_SPEC void sub(const double * pSrc1, const double * pSrc2, double * pDst, int len)
	{
		internals::dPtrPtrDst<_mm_sub_pd, _mm_sub_sd>(pSrc1, pSrc2, pDst, len);
	}

	_SIMD_SSE_SPEC void mul(const double * pSrc1, const double * pSrc2, double * pDst, int len)
	{
		internals::dPtrPtrDst<_mm_mul_pd, _mm_mul_sd>(pSrc1, pSrc2, pDst, len);
	}

	_SIMD_SSE_SPEC void div(const double * pSrc1, const double * pSrc2, double * pDst, int len)
	{
		internals::dPtrPtrDst<_mm_div_pd, _mm_div_sd>(pSrc1, pSrc2, pDst, len);
	}

	_SIMD_SSE_SPEC void abs(const double * pSrc, double * pDst, int len)
	{
		internals::dPtrDst<internals::abs_pd, internals::abs_pd>(pSrc, pDst, len);
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
		internals::dPtrDst<_mm_sqrt_pd, _mm_sqrt_pd>(pSrc, pDst, len);
	}

	_SIMD_SSE_SPEC void invSqrt(const double * pSrc, double * pDst, int len)
	{
		sqrt(pSrc, pDst, len);
		inv(pDst, pDst, len);
	}
}

// TODO
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
