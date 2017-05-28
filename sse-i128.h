#pragma once
#include <cstdint>
#include <smmintrin.h>

#include "sse.h"

namespace sse
{
namespace internals
{
	template <	IntrI::Unary op,
				IntrI::Store store = xx_store_si>
	INLINE void iValDst(__m128i * pDst, __m128i&& a, int len)
	{
		for (; len >= 4; len-=4, pDst+=4)
		{
			store(pDst, op(a));
			store(pDst+1, op(a));
			store(pDst+2, op(a));
			store(pDst+3, op(a));
		}

		if (len >= 2)
		{
			store(pDst, op(a));
			store(pDst+1, op(a));
			len -= 2; pDst += 2;
		}

		if (len)
		{
			store(pDst, op(a));
		}
	}

	template <	IntrI::Unary op,
				IntrI::Load load = xx_load_si,
				IntrI::Store store = xx_store_si>
	INLINE void iPtrDst(const __m128i * pSrc, __m128i * pDst, int len)
	{
		for (; len >= 4; len-=4, pSrc+=4, pDst+=4)
		{
			store(pDst, op(load(pSrc)));
			store(pDst+1, op(load(pSrc+1)));
			store(pDst+2, op(load(pSrc+2)));
			store(pDst+3, op(load(pSrc+3)));
		}

		if (len >= 2)
		{
			store(pDst, op(load(pSrc)));
			store(pDst+1, op(load(pSrc+1)));

			len -= 2; pSrc += 2; pDst += 2;
		}

		if (len)
		{
			store(pDst, op(load(pSrc)));
		}
	}

	template <	IntrI::Binary op,
				IntrI::Load load = xx_load_si,
				IntrI::Store store = xx_store_si>
	INLINE void iPtrValDst(const __m128i * pSrc, __m128i&& b, __m128i * pDst, int len)
	{
		for (; len >= 4; len-=4, pSrc+=4, pDst+=4)
		{
			store(pDst, op(load(pSrc), b));
			store(pDst+1, op(load(pSrc+1), b));
			store(pDst+2, op(load(pSrc+2), b));
			store(pDst+3, op(load(pSrc+3), b));
		}

		if (len >= 2)
		{
			store(pDst, op(load(pSrc), b));
			store(pDst+1, op(load(pSrc+1), b));

			len -= 2; pSrc += 2; pDst += 2;
		}

		if (len)
		{
			store(pDst, op(load(pSrc), b));
		}
	}

	template <	IntrI::Binary op,
				IntrI::Load load = xx_load_si,
				IntrI::Store store = xx_store_si>
	INLINE void iPtrPtrDst(const __m128i * pSrc1, const __m128i * pSrc2, __m128i * pDst, int len)
	{
		for (; len >= 4; len-=4, pSrc1+=4, pSrc2+=4, pDst+=4)
		{
			__m128i a0 = load(pSrc1);
			__m128i a1 = load(pSrc1+1);
			__m128i a2 = load(pSrc1+2);
			__m128i a3 = load(pSrc1+3);

			__m128i b0 = load(pSrc2);
			__m128i b1 = load(pSrc2+1);
			__m128i b2 = load(pSrc2+2);
			__m128i b3 = load(pSrc2+3);

			store(pDst, op(a0, b0));
			store(pDst+1, op(a1, b1));
			store(pDst+2, op(a2, b2));
			store(pDst+3, op(a3, b3));
		}

		if (len >= 2)
		{
			store(pDst, op(load(pSrc1), load(pSrc2)));
			store(pDst+1, op(load(pSrc1+1), load(pSrc2+1)));

			len -= 2; pSrc1 += 2; pSrc2 += 2; pDst += 2;
		}

		if (len)
		{
			store(pDst, op(load(pSrc1), load(pSrc2)));
		}
	}
}

namespace common
{
	_SIMD_SSE_SPEC void set(int32_t val, int32_t * pDst, int len)
	{
		internals::iValDst<nop>((__m128i*)pDst, _mm_set1_epi32(val), (len>>2));
		for (int i = len - (len & 0x3); i < len; ++i)
			pDst[i] = val;
	}

	_SIMD_SSE_SPEC void set(uint32_t val, uint32_t * pDst, int len)
	{
		set((int32_t)val, (int32_t*)pDst, len);
	}

	_SIMD_SSE_SPEC void set(int64_t val, int64_t * pDst, int len)
	{
		internals::iValDst<nop>((__m128i*)pDst, _mm_set1_epi64x(val), (len>>1));
		if (len)
			pDst[len-1] = val;
	}

	_SIMD_SSE_SPEC void set(uint64_t val, uint64_t * pDst, int len)
	{
		set((int64_t)val, (int64_t*)pDst, len);
	}

	_SIMD_SSE_SPEC void copy(const int32_t * pSrc, int32_t * pDst, int len)
	{
		internals::iPtrDst<nop>((const __m128i*)pSrc, (__m128i*)pDst, (len>>2));
		for (int i = len - (len & 0x3); i < len; ++i)
			pDst[i] = pSrc[i];
	}

	_SIMD_SSE_SPEC void copy(const uint32_t * pSrc, uint32_t * pDst, int len)
	{
		copy((const int32_t*)pSrc, (int32_t*)pDst, len);
	}

	_SIMD_SSE_SPEC void copy(const int64_t * pSrc, int64_t * pDst, int len)
	{
		internals::iPtrDst<nop>((const __m128i*)pSrc, (__m128i*)pDst, (len>>1));
		if (len)
			pDst[len-1] = pSrc[len-1];
	}

	_SIMD_SSE_SPEC void copy(const uint64_t * pSrc, uint64_t * pDst, int len)
	{
		copy((const int64_t*)pSrc, (int64_t*)pDst, len);
	}
}

namespace arithmetic
{
	_SIMD_SSE_SPEC void addC(const int32_t * pSrc, int32_t val, int32_t * pDst, int len)
	{
		internals::iPtrValDst<_mm_add_epi32>((const __m128i*)pSrc, _mm_set1_epi32(val), (__m128i*)pDst, (len>>2));
		for (int i = len - (len & 0x3); i < len; ++i)
			pDst[i] = pSrc[i] + val;
	}

	_SIMD_SSE_SPEC void subC(const int32_t * pSrc, int32_t val, int32_t * pDst, int len)
	{
		internals::iPtrValDst<_mm_sub_epi32>((const __m128i*)pSrc, _mm_set1_epi32(val), (__m128i*)pDst, (len>>2));
		for (int i = len - (len & 0x3); i < len; ++i)
			pDst[i] = pSrc[i] - val;
	}

	_SIMD_SSE_SPEC void subCRev(const int32_t * pSrc, int32_t val, int32_t * pDst, int len)
	{
		internals::iPtrValDst<IntrI::rev_op<_mm_sub_epi32>>((const __m128i*)pSrc, _mm_set1_epi32(val), (__m128i*)pDst, (len>>2));
		for (int i = len - (len & 0x3); i < len; ++i)
			pDst[i] = val - pSrc[i];
	}

	_SIMD_SSE_SPEC void mulC(const int32_t * pSrc, int32_t val, int32_t * pDst, int len)
	{
		internals::iPtrValDst<_mm_mullo_epi32>((const __m128i*)pSrc, _mm_set1_epi32(val), (__m128i*)pDst, (len>>2));
		for (int i = len - (len & 0x3); i < len; ++i)
			pDst[i] = pSrc[i] * val;
	}

	_SIMD_SSE_SPEC void divC(const int32_t * pSrc, int32_t val, int32_t * pDst, int len)
	{
		nosimd::arithmetic::divC(pSrc, val, pDst, len);
	}

	_SIMD_SSE_SPEC void divCRev(const int32_t * pSrc, int32_t val, int32_t * pDst, int len)
	{
		nosimd::arithmetic::divCRev(pSrc, val, pDst, len);
	}


	_SIMD_SSE_SPEC void add(const int32_t * pSrc1, const int32_t * pSrc2, int32_t * pDst, int len)
	{
		internals::iPtrPtrDst<_mm_add_epi32>((const __m128i*)pSrc1, (const __m128i*)pSrc2, (__m128i*)pDst, (len>>2));
		for (int i = len - (len & 0x3); i < len; ++i)
			pDst[i] = pSrc1[i] + pSrc2[i];
	}

	_SIMD_SSE_SPEC void sub(const int32_t * pSrc1, const int32_t * pSrc2, int32_t * pDst, int len)
	{
		internals::iPtrPtrDst<_mm_sub_epi32>((const __m128i*)pSrc1, (const __m128i*)pSrc2, (__m128i*)pDst, (len>>2));
		for (int i = len - (len & 0x3); i < len; ++i)
			pDst[i] = pSrc1[i] - pSrc2[i];
	}

	_SIMD_SSE_SPEC void mul(const int32_t * pSrc1, const int32_t * pSrc2, int32_t * pDst, int len)
	{
		internals::iPtrPtrDst<_mm_mullo_epi32>((const __m128i*)pSrc1, (const __m128i*)pSrc2, (__m128i*)pDst, (len>>2));
		for (int i = len - (len & 0x3); i < len; ++i)
			pDst[i] = pSrc1[i] * pSrc2[i];
	}

	_SIMD_SSE_SPEC void div(const int32_t * pSrc1, const int32_t * pSrc2, int32_t * pDst, int len)
	{
		nosimd::arithmetic::div(pSrc1, pSrc2, pDst, len);
	}

	_SIMD_SSE_SPEC void abs(const int32_t * pSrc, int32_t * pDst, int len)
	{
		internals::iPtrDst<_mm_abs_epi32>((const __m128i*)pSrc, (__m128i*)pDst, (len>>2));
		for (int i = len - (len & 0x3); i < len; ++i)
			pDst[i] = (pSrc[i] > 0) ? pSrc[i] : (-pSrc[i]);
	}

	//

	_SIMD_SSE_SPEC void addC(const uint32_t * pSrc, uint32_t val, uint32_t * pDst, int len)
	{
		addC((const int32_t*)pSrc, (int32_t)val, (int32_t*)pDst, len);
	}

	_SIMD_SSE_SPEC void subC(const uint32_t * pSrc, uint32_t val, uint32_t * pDst, int len)
	{
		subC((const int32_t*)pSrc, (int32_t)val, (int32_t*)pDst, len);
	}

	_SIMD_SSE_SPEC void subCRev(const uint32_t * pSrc, uint32_t val, uint32_t * pDst, int len)
	{
		subCRev((const int32_t*)pSrc, (int32_t)val, (int32_t*)pDst, len);
	}

	_SIMD_SSE_SPEC void mulC(const uint32_t * pSrc, uint32_t val, uint32_t * pDst, int len)
	{
#if 1
		mulC((const int32_t*)pSrc, (int32_t)val, (int32_t*)pDst, len);
#else
		nosimd::arithmetic::mulC(pSrc, val, pDst, len);
#endif
	}

	_SIMD_SSE_SPEC void divC(const uint32_t * pSrc, uint32_t val, uint32_t * pDst, int len)
	{
		nosimd::arithmetic::divC(pSrc, val, pDst, len);
	}

	_SIMD_SSE_SPEC void divCRev(const uint32_t * pSrc, uint32_t val, uint32_t * pDst, int len)
	{
		nosimd::arithmetic::divCRev(pSrc, val, pDst, len);
	}


	_SIMD_SSE_SPEC void add(const uint32_t * pSrc1, const uint32_t * pSrc2, uint32_t * pDst, int len)
	{
		add((const int32_t*)pSrc1, (const int32_t*)pSrc2, (int32_t*)pDst, len);
	}

	_SIMD_SSE_SPEC void sub(const uint32_t * pSrc1, const uint32_t * pSrc2, uint32_t * pDst, int len)
	{
		sub((const int32_t*)pSrc1, (const int32_t*)pSrc2, (int32_t*)pDst, len);
	}

	_SIMD_SSE_SPEC void mul(const uint32_t * pSrc1, const uint32_t * pSrc2, uint32_t * pDst, int len)
	{
#if 1
		mul((const int32_t*)pSrc1, (const int32_t*)pSrc2, (int32_t*)pDst, len);
#else
		nosimd::arithmetic::mul(pSrc1, pSrc2, pDst, len);
#endif
	}

	_SIMD_SSE_SPEC void div(const uint32_t * pSrc1, const uint32_t * pSrc2, uint32_t * pDst, int len)
	{
		nosimd::arithmetic::div(pSrc1, pSrc2, pDst, len);
	}

	_SIMD_SSE_SPEC void abs(const uint32_t * pSrc, uint32_t * pDst, int len)
	{
		if (pSrc != pDst)
			copy(pSrc, pDst, len);
	}

	//

	_SIMD_SSE_SPEC void addC(const int64_t * pSrc, int64_t val, int64_t * pDst, int len)
	{
		internals::iPtrValDst<_mm_add_epi64>((const __m128i*)pSrc, _mm_set1_epi64x(val), (__m128i*)pDst, (len>>1));
		if (len)
			pDst[len-1] = pSrc[len-1] + val;
	}

	_SIMD_SSE_SPEC void subC(const int64_t * pSrc, int64_t val, int64_t * pDst, int len)
	{
		internals::iPtrValDst<_mm_sub_epi64>((const __m128i*)pSrc, _mm_set1_epi64x(val), (__m128i*)pDst, (len>>1));
		if (len)
			pDst[len-1] = pSrc[len-1] - val;
	}

	_SIMD_SSE_SPEC void subCRev(const int64_t * pSrc, int64_t val, int64_t * pDst, int len)
	{
		internals::iPtrValDst<IntrI::rev_op<_mm_sub_epi64>>((const __m128i*)pSrc, _mm_set1_epi64x(val), (__m128i*)pDst, (len>>1));
		if (len)
			pDst[len-1] = val - pSrc[len-1];
	}

	_SIMD_SSE_SPEC void mulC(const int64_t * pSrc, int64_t val, int64_t * pDst, int len)
	{
		nosimd::arithmetic::mulC(pSrc, val, pDst, len);
	}

	_SIMD_SSE_SPEC void divC(const int64_t * pSrc, int64_t val, int64_t * pDst, int len)
	{
		nosimd::arithmetic::divC(pSrc, val, pDst, len);
	}

	_SIMD_SSE_SPEC void divCRev(const int64_t * pSrc, int64_t val, int64_t * pDst, int len)
	{
		nosimd::arithmetic::divCRev(pSrc, val, pDst, len);
	}


	_SIMD_SSE_SPEC void add(const int64_t * pSrc1, const int64_t * pSrc2, int64_t * pDst, int len)
	{
		internals::iPtrPtrDst<_mm_add_epi64>((const __m128i*)pSrc1, (const __m128i*)pSrc2, (__m128i*)pDst, (len>>1));
		if (len)
			pDst[len-1] = pSrc1[len-1] + pSrc2[len-1];
	}

	_SIMD_SSE_SPEC void sub(const int64_t * pSrc1, const int64_t * pSrc2, int64_t * pDst, int len)
	{
		internals::iPtrPtrDst<_mm_sub_epi64>((const __m128i*)pSrc1, (const __m128i*)pSrc2, (__m128i*)pDst, (len>>1));
		if (len)
			pDst[len-1] = pSrc1[len-1] - pSrc2[len-1];
	}

	_SIMD_SSE_SPEC void mul(const int64_t * pSrc1, const int64_t * pSrc2, int64_t * pDst, int len)
	{
		nosimd::arithmetic::mul(pSrc1, pSrc2, pDst, len);
	}

	_SIMD_SSE_SPEC void div(const int64_t * pSrc1, const int64_t * pSrc2, int64_t * pDst, int len)
	{
		nosimd::arithmetic::div(pSrc1, pSrc2, pDst, len);
	}

	_SIMD_SSE_SPEC void abs(const int64_t * pSrc, int64_t * pDst, int len)
	{
		nosimd::arithmetic::abs(pSrc, pDst, len);
	}

	//

	_SIMD_SSE_SPEC void addC(const uint64_t * pSrc, uint64_t val, uint64_t * pDst, int len)
	{
		addC((const int64_t*)pSrc, (int64_t)val, (int64_t*)pDst, len);
	}

	_SIMD_SSE_SPEC void subC(const uint64_t * pSrc, uint64_t val, uint64_t * pDst, int len)
	{
		subC((const int64_t*)pSrc, (int64_t)val, (int64_t*)pDst, len);
	}

	_SIMD_SSE_SPEC void subCRev(const uint64_t * pSrc, uint64_t val, uint64_t * pDst, int len)
	{
		subCRev((const int64_t*)pSrc, (int64_t)val, (int64_t*)pDst, len);
	}

	_SIMD_SSE_SPEC void mulC(const uint64_t * pSrc, uint64_t val, uint64_t * pDst, int len)
	{
		nosimd::arithmetic::mulC(pSrc, val, pDst, len);
	}

	_SIMD_SSE_SPEC void divC(const uint64_t * pSrc, uint64_t val, uint64_t * pDst, int len)
	{
		nosimd::arithmetic::divC(pSrc, val, pDst, len);
	}

	_SIMD_SSE_SPEC void divCRev(const uint64_t * pSrc, uint64_t val, uint64_t * pDst, int len)
	{
		nosimd::arithmetic::divCRev(pSrc, val, pDst, len);
	}


	_SIMD_SSE_SPEC void add(const uint64_t * pSrc1, const uint64_t * pSrc2, uint64_t * pDst, int len)
	{
		add((const int64_t*)pSrc1, (const int64_t*)pSrc2, (int64_t*)pDst, len);
	}

	_SIMD_SSE_SPEC void sub(const uint64_t * pSrc1, const uint64_t * pSrc2, uint64_t * pDst, int len)
	{
		sub((const int64_t*)pSrc1, (const int64_t*)pSrc2, (int64_t*)pDst, len);
	}

	_SIMD_SSE_SPEC void mul(const uint64_t * pSrc1, const uint64_t * pSrc2, uint64_t * pDst, int len)
	{
		nosimd::arithmetic::mul(pSrc1, pSrc2, pDst, len);
	}

	_SIMD_SSE_SPEC void div(const uint64_t * pSrc1, const uint64_t * pSrc2, uint64_t * pDst, int len)
	{
		nosimd::arithmetic::div(pSrc1, pSrc2, pDst, len);
	}

	_SIMD_SSE_SPEC void abs(const uint64_t * pSrc, uint64_t * pDst, int len)
	{
		if (pSrc != pDst)
			copy(pSrc, pDst, len);
	}
}

namespace power
{
	// TODO
}

namespace statistical
{
	// TODO
}
}
