#pragma once
#include <cstdint>
#include <smmintrin.h>

#include "sse.h"

namespace sse
{
namespace internals
{
	INLINE __m128 abs_ps(__m128 x)
	{
		static const __m128 sign_mask = _mm_set1_ps(-0.f); // -0.f = 1 << 31
		return _mm_andnot_ps(sign_mask, x); // !sign_mask & x
	}

	// TODO: replace with _mm_hadd_ps?
	INLINE __m128 horizontal_sum(__m128 x)
	{
		// some magic here
		const __m128 t = _mm_add_ps(x, _mm_movehl_ps(x, x));
		return _mm_add_ss(t, _mm_shuffle_ps(t, t, 1));
	}

	//

	template <	IntrS::Unary op_ps,
				IntrS::Unary op_ss,
				IntrS::Load load_ps = xx_load_ps,
				IntrS::Store store_ps = xx_store_ps>
	INLINE void sPtrDst(const float * pSrc, float * pDst, int len)
	{
#ifdef UNROLL_MORE
		for (; len >= 16; len-=16, pSrc+=16, pDst+=16)
		{
			__m128 a0 = load_ps(pSrc);
			__m128 a1 = load_ps(pSrc+4);
			__m128 a2 = load_ps(pSrc+8);
			__m128 a3 = load_ps(pSrc+12);

			a0 = op_ps(a0);
			a1 = op_ps(a1);
			a2 = op_ps(a2);
			a3 = op_ps(a3);

			store_ps(pDst, a0);
			store_ps(pDst+4, a1);
			store_ps(pDst+8, a2);
			store_ps(pDst+12, a3);
		}
#endif
		for (; len >= 8; len-=8, pSrc+=8, pDst+=8)
		{
			__m128 a0 = load_ps(pSrc);
			__m128 a1 = load_ps(pSrc+4);

			a0 = op_ps(a0);
			a1 = op_ps(a1);

			store_ps(pDst, a0);
			store_ps(pDst+4, a1);
		}

		if (len >= 4)
		{
			__m128 a0 = load_ps(pSrc);
			a0 = op_ps(a0);
			store_ps(pDst, a0);

			len -= 4; pSrc += 4; pDst += 4;
		}

		for (; len > 0; --len, ++pSrc, ++pDst)
		{
			__m128 a0 = _mm_load_ss(pSrc);
			a0 = op_ps(a0);
			_mm_store_ss(pDst, a0);
		}
	}

	template <	IntrS::Binary op_ps,
				IntrS::Binary op_ss,
				IntrS::Load load_ps = xx_load_ps,
				IntrS::Store store_ps = xx_store_ps>
	INLINE void sPtrValDst(const float * pSrc, float val, float * pDst, int len)
	{
		const __m128 b = _mm_set1_ps(val);
#ifdef UNROLL_MORE
		for (; len >= 16; len-=16, pSrc+=16, pDst+=16)
		{
			__m128 a0 = load_ps(pSrc);
			__m128 a1 = load_ps(pSrc+4);
			__m128 a2 = load_ps(pSrc+8);
			__m128 a3 = load_ps(pSrc+12);

			a0 = op_ps(a0, b);
			a1 = op_ps(a1, b);
			a2 = op_ps(a2, b);
			a3 = op_ps(a3, b);

			store_ps(pDst, a0);
			store_ps(pDst+4, a1);
			store_ps(pDst+8, a2);
			store_ps(pDst+12, a3);
		}
#endif
		for (; len >= 8; len-=8, pSrc+=8, pDst+=8)
		{
			__m128 a0 = load_ps(pSrc);
			__m128 a1 = load_ps(pSrc+4);

			a0 = op_ps(a0, b);
			a1 = op_ps(a1, b);

			store_ps(pDst, a0);
			store_ps(pDst+4, a1);
		}

		if (len >= 4)
		{
			__m128 a0 = load_ps(pSrc);
			a0 = op_ps(a0, b);
			store_ps(pDst, a0);

			len -= 4; pSrc += 4; pDst += 4;
		}

		for (; len > 0; --len, ++pSrc, ++pDst)
		{
			__m128 a0 = _mm_load_ss(pSrc);
			a0 = op_ss(a0, b);
			_mm_store_ss(pDst, a0);
		}
	}

	template <	IntrS::Binary op_ps,
				IntrS::Binary op_ss,
				IntrS::Load load_ps = xx_load_ps,
				IntrS::Store store_ps = xx_store_ps>
	INLINE void sPtrPtrDst(const float * pSrc1, const float * pSrc2, float * pDst, int len)
	{
#ifdef UNROLL_MORE
		for (; len >= 16; len-=16, pSrc1+=16, pSrc2+=16, pDst+=16)
		{
			__m128 a0 = load_ps(pSrc1);
			__m128 a1 = load_ps(pSrc1+4);
			__m128 a2 = load_ps(pSrc1+8);
			__m128 a3 = load_ps(pSrc1+12);

			__m128 b0 = load_ps(pSrc2);
			__m128 b1 = load_ps(pSrc2+4);
			__m128 b2 = load_ps(pSrc2+8);
			__m128 b3 = load_ps(pSrc2+12);

			a0 = op_ps(a0, b0);
			a1 = op_ps(a1, b1);
			a2 = op_ps(a2, b2);
			a3 = op_ps(a3, b3);

			store_ps(pDst, a0);
			store_ps(pDst+4, a1);
			store_ps(pDst+8, a2);
			store_ps(pDst+12, a3);
		}
#endif
		for (; len >= 8; len-=8, pSrc1+=8, pSrc2+=8, pDst+=8)
		{
			__m128 a0 = load_ps(pSrc1);
			__m128 a1 = load_ps(pSrc1+4);

			__m128 b0 = load_ps(pSrc2);
			__m128 b1 = load_ps(pSrc2+4);

			a0 = op_ps(a0, b0);
			a1 = op_ps(a1, b1);

			store_ps(pDst, a0);
			store_ps(pDst+4, a1);
		}

		if (len >= 4)
		{
			__m128 a0 = load_ps(pSrc1);
			__m128 b0 = load_ps(pSrc2);

			a0 = op_ps(a0, b0);
			store_ps(pDst, a0);

			len -= 4; pSrc1 += 4; pSrc2 += 4; pDst += 4;
		}

		for (; len > 0; --len, ++pSrc1, ++pSrc2, ++pDst)
		{
			__m128 a0 = _mm_load_ss(pSrc1);
			__m128 b0 = _mm_load_ss(pSrc2);

			a0 = op_ss(a0, b0);
			_mm_store_ss(pDst, a0);
		}
	}

	template <	IntrS::Binary op_ps,
				IntrS::Binary op_ss,
				IntrS::Load load_one,
				IntrS::Load load_ps = xx_load_ps>
	INLINE void aggregate(const float * pSrc, int len, __m128& r0)
	{
		if (len >= 4)
		{
			r0 = load_ps(pSrc);
			len -= 4; pSrc += 4;

			if (len >= 8)
			{
				__m128 r1 = load_ps(pSrc);
				len -= 4; pSrc += 4;

				if (len >= 16)
				{
					__m128 r2 = load_ps(pSrc);
					__m128 r3 = load_ps(pSrc+4);
					len -= 8; pSrc += 8;

					for (; len >= 16; len-=16, pSrc+=16)
					{
						__m128 a0 = load_ps(pSrc);
						__m128 a1 = load_ps(pSrc+4);
						__m128 a2 = load_ps(pSrc+8);
						__m128 a3 = load_ps(pSrc+12);

						r0 = op_ps(r0, a0);
						r1 = op_ps(r1, a1);
						r2 = op_ps(r2, a2);
						r3 = op_ps(r3, a3);
					}

					r0 = op_ps(r0, r2);
					r1 = op_ps(r1, r3);
				}

				if (len >= 8)
				{
					__m128 a0 = load_ps(pSrc);
					__m128 a1 = load_ps(pSrc+4);

					r0 = op_ps(r0, a0);
					r1 = op_ps(r1, a1);

					len -=8; pSrc += 8;
				}

				r0 = op_ps(r0, r1);
			}

			if (len >= 4)
			{
				__m128 a0 = load_ps(pSrc);
				r0 = op_ps(r0, a0);
				len -= 4; pSrc += 4;
			}
		}
		else
		{
			r0 = load_one(pSrc);
			--len; ++pSrc;
		}

		for (; len; --len, ++pSrc)
		{
			__m128 a0 = _mm_load_ss(pSrc);
			r0 = op_ss(r0, a0);
		}
	}
}

namespace common
{
	template <IntrS::Store store_ps = xx_store_ps>
	INLINE void setT(float val, float * pDst, int len)
	{
		__m128 a = _mm_set1_ps(val);

		for (; len >= 16; len-=16, pDst+=16)
		{
			store_ps(pDst, a);
			store_ps(pDst+4, a);
			store_ps(pDst+8, a);
			store_ps(pDst+12, a);
		}

		if (len >= 8)
		{
			store_ps(pDst, a);
			store_ps(pDst+4, a);
			len -= 8; pDst += 8;
		}

		if (len >= 4)
		{
			store_ps(pDst, a);
			len -= 4; pDst += 4;
		}

		for (; len > 0; --len, ++pDst)
			*pDst = val;
	}

	_SIMD_SSE_SPEC void set(float val, float * pDst, int len)
	{
		setT(val, pDst, len);
	}

	_SIMD_SSE_SPEC void copy(const float * pSrc, float * pDst, int len)
	{
		internals::sPtrDst<nop, nop>(pSrc, pDst, len);
	}
}

namespace arithmetic
{
	_SIMD_SSE_SPEC void addC(const float * pSrc, float val, float * pDst, int len)
	{
		internals::sPtrValDst<_mm_add_ps, _mm_add_ss>(pSrc, val, pDst, len);
	}

	_SIMD_SSE_SPEC void subC(const float * pSrc, float val, float * pDst, int len)
	{
		internals::sPtrValDst<_mm_sub_ps, _mm_sub_ss>(pSrc, val, pDst, len);
	}

	_SIMD_SSE_SPEC void mulC(const float * pSrc, float val, float * pDst, int len)
	{
		internals::sPtrValDst<_mm_mul_ps, _mm_mul_ss>(pSrc, val, pDst, len);
	}

	_SIMD_SSE_SPEC void divC(const float * pSrc, float val, float * pDst, int len)
	{
		internals::sPtrValDst<_mm_div_ps, _mm_div_ss>(pSrc, val, pDst, len);
	}


	_SIMD_SSE_SPEC void subCRev(const float * pSrc, float val, float * pDst, int len)
	{
		internals::sPtrValDst<IntrS::rev_op<_mm_sub_ps>, IntrS::rev_op<_mm_sub_ss>>(pSrc, val, pDst, len);
	}

	_SIMD_SSE_SPEC void divCRev(const float * pSrc, float val, float * pDst, int len)
	{
		internals::sPtrValDst<IntrS::rev_op<_mm_div_ps>, IntrS::rev_op<_mm_div_ss>>(pSrc, val, pDst, len);
	}


	_SIMD_SSE_SPEC void add(const float * pSrc1, const float * pSrc2, float * pDst, int len)
	{
		internals::sPtrPtrDst<_mm_add_ps, _mm_add_ss>(pSrc1, pSrc2, pDst, len);
	}

	_SIMD_SSE_SPEC void sub(const float * pSrc1, const float * pSrc2, float * pDst, int len)
	{
		internals::sPtrPtrDst<_mm_sub_ps, _mm_sub_ss>(pSrc1, pSrc2, pDst, len);
	}

	_SIMD_SSE_SPEC void mul(const float * pSrc1, const float * pSrc2, float * pDst, int len)
	{
		internals::sPtrPtrDst<_mm_mul_ps, _mm_mul_ss>(pSrc1, pSrc2, pDst, len);
	}

	_SIMD_SSE_SPEC void div(const float * pSrc1, const float * pSrc2, float * pDst, int len)
	{
		internals::sPtrPtrDst<_mm_div_ps, _mm_div_ss>(pSrc1, pSrc2, pDst, len);
	}

	_SIMD_SSE_SPEC void abs(const float * pSrc, float * pDst, int len)
	{
		internals::sPtrDst<internals::abs_ps, internals::abs_ps>(pSrc, pDst, len);
	}
}

namespace power
{
	_SIMD_SSE_SPEC void inv(const float * pSrc, float * pDst, int len)
	{
		internals::sPtrDst<_mm_rcp_ps, _mm_rcp_ss>(pSrc, pDst, len);
	}

	_SIMD_SSE_SPEC void sqrt(const float * pSrc, float * pDst, int len)
	{
		internals::sPtrDst<_mm_sqrt_ps, _mm_sqrt_ss>(pSrc, pDst, len);
	}

	_SIMD_SSE_SPEC void invSqrt(const float * pSrc, float * pDst, int len)
	{
		internals::sPtrDst<_mm_rsqrt_ps, _mm_rsqrt_ss>(pSrc, pDst, len);
	}
}

namespace statistical
{
	_SIMD_SSE_SPEC void min(const float * pSrc, int len, float * pMin)
	{
		__m128 r0;
		float res[4];
		internals::aggregate<_mm_min_ps, _mm_min_ss, _mm_load1_ps>(pSrc, len, r0);
		_mm_storeu_ps(res, r0);

		res[0] = (res[0] < res[1]) ? res[0] : res[1];
		res[2] = (res[2] < res[3]) ? res[2] : res[3];
		*pMin = (res[0] < res[2]) ? res[0] : res[2];
	}

	_SIMD_SSE_SPEC void max(const float * pSrc, int len, float * pMin)
	{
		__m128 r0;
		float res[4];
		internals::aggregate<_mm_max_ps, _mm_max_ss, _mm_load1_ps>(pSrc, len, r0);
		_mm_storeu_ps(res, r0);

		res[0] = (res[0] > res[1]) ? res[0] : res[1];
		res[2] = (res[2] > res[3]) ? res[2] : res[3];
		*pMin = (res[0] > res[2]) ? res[0] : res[2];
	}

	_SIMD_SSE_SPEC void minMax(const float * pSrc, int len, float * pMin, float * pMax)
	{
		return nosimd::statistical::minMax(pSrc, len, pMin, pMax);
	}

	_SIMD_SSE_SPEC void sum(const float * pSrc, int len, float * pSum)
	{
		__m128 r0 = _mm_setzero_ps();
		internals::aggregate<_mm_add_ps, _mm_add_ss, _mm_load_ss>(pSrc, len, r0);

		r0 = internals::horizontal_sum(r0);
		*pSum = _mm_cvtss_f32(r0);
	}

	template <IntrS::Load load_ps = xx_load_ps>
	INLINE void meanStdDevT(const float * pSrc, int len, float * pMean, float * pStdDev)
	{
		const float coef = len-1;
		mean(pSrc, len, pMean);

		const __m128 b = _mm_load1_ps(pMean);
		__m128 r0 = _mm_setzero_ps();
		__m128 r1 = _mm_setzero_ps();

		for (; len >= 8; len-=8, pSrc+=8)
		{
			__m128 a0 = load_ps(pSrc);
			__m128 a1 = load_ps(pSrc+4);

			a0 = _mm_sub_ps(a0, b);
			a1 = _mm_sub_ps(a1, b);

			a0 = _mm_mul_ps(a0, a0);
			a1 = _mm_mul_ps(a1, a1);

			r0 = _mm_add_ps(r0, a0);
			r1 = _mm_add_ps(r1, a1);
		}

		r0 = _mm_add_ps(r0, r1);

		if (len >= 4)
		{
			__m128 a0 = load_ps(pSrc);
			a0 = _mm_sub_ps(a0, b);
			a0 = _mm_mul_ps(a0, a0);
			r0 = _mm_add_ps(r0, a0);

			len -= 4; pSrc += 4;
		}

		for (; len; --len, ++pSrc)
		{
			__m128 a0 = _mm_load_ss(pSrc);
			a0 = _mm_sub_ss(a0, b);
			a0 = _mm_mul_ss(a0, a0);

			r0 = _mm_add_ss(r0, a0);
		}

		const __m128 t = _mm_add_ps(r0, _mm_movehl_ps(r0, r0));
		r1 = _mm_add_ss(t, _mm_shuffle_ps(t, t, 1));

		r0 = _mm_set1_ps(coef);
		r1 = _mm_div_ss(r1, r0);
		r1 = _mm_sqrt_ss(r1);

		*pStdDev = _mm_cvtss_f32(r1);
	}

	_SIMD_SSE_SPEC void meanStdDev(const float * pSrc, int len, float * pMean, float * pStdDev)
	{
		meanStdDevT(pSrc, len, pMean, pStdDev);
	}

	// TODO: perf tests
	template <IntrS::Load load_ps = xx_load_ps>
	INLINE void dotProd_v1(const float * pSrc1, const float * pSrc2, int len, float * pDp)
	{
		__m128 r0 = _mm_setzero_ps();
		__m128 r1 = _mm_setzero_ps();

		for (; len >= 8; len-=8, pSrc1+=8, pSrc2+=8)
		{
			__m128 a0 = load_ps(pSrc1);
			__m128 a1 = load_ps(pSrc1+4);

			__m128 b0 = load_ps(pSrc2);
			__m128 b1 = load_ps(pSrc2+4);

			r0 = _mm_add_ps(r0, _mm_mul_ps(a0, b0));
			r1 = _mm_add_ps(r1, _mm_mul_ps(a1, b1));
		}

		r0 = _mm_add_ps(r0, r1);

		if (len >= 4)
		{
			__m128 a0 = load_ps(pSrc1);
			__m128 b0 = load_ps(pSrc2);

			r0 = _mm_add_ps(r0, _mm_mul_ps(a0, b0));

			len -= 4; pSrc1 += 4; pSrc2 += 4;
		}

		for (; len; --len, ++pSrc1, ++pSrc2)
		{
			__m128 a0 = _mm_load_ss(pSrc1);
			__m128 b0 = _mm_load_ss(pSrc2);

			r0 = _mm_add_ss(r0, _mm_mul_ss(a0, b0));
		}

		r0 = internals::horizontal_sum(r0);
		*pDp = _mm_cvtss_f32(r0);
	}

	template <IntrS::Load load_ps = xx_load_ps>
	INLINE void dotProd_v2(const float * pSrc1, const float * pSrc2, int len, float * pDp)
	{
		__m128 dp = _mm_setzero_ps();

		for (; len >= 8; len-=8, pSrc1+=8, pSrc2+=8)
		{
			__m128 a0 = load_ps(pSrc1);
			__m128 a1 = load_ps(pSrc1+4);
			__m128 b0 = load_ps(pSrc2);
			__m128 b1 = load_ps(pSrc2+4);

			__m128 sum = _mm_add_ss(_mm_dp_ps(a0, b0, 0xf1), _mm_dp_ps(a1, b1, 0xf1));
			dp = _mm_add_ss(dp, sum);
		}

		if (len >= 4)
		{
			__m128 a0 = load_ps(pSrc1);
			__m128 b0 = load_ps(pSrc2);
			dp = _mm_add_ss(dp, _mm_dp_ps(a0, b0, 0xf1));

			len -= 4; pSrc1 += 4; pSrc2 += 4;
		}

		for (; len; --len, ++pSrc1, ++pSrc2)
		{
			__m128 a0 = _mm_load_ss(pSrc1);
			__m128 b0 = _mm_load_ss(pSrc2);
			dp = _mm_add_ss(dp, _mm_mul_ss(a0, b0));
		}

		*pDp = _mm_cvtss_f32(dp);
	}

	_SIMD_SSE_SPEC void dotProd(const float * pSrc1, const float * pSrc2, int len, float * pDp)
	{
		dotProd_v1(pSrc1, pSrc2, len, pDp);
	}
}
}
