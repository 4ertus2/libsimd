#pragma once
#include <cstdint>
#include <smmintrin.h>

#include "sse.h"

namespace sse_i128_internal
{
	typedef __m128i (*IntrI)(__m128i);
	typedef __m128i (*IntrII)(__m128i, __m128i);
	typedef __m128i (*IntrPi)(const __m128i *);
	typedef void (*IntrPI)(__m128i*, __m128i);

#ifdef SSE_ALIGNED
	INLINE __m128i xx_load_si128(const __m128i * x) { return _mm_load_si128(x); }
	//INLINE __m128i xx_load_si128(const __m128i * x) { return _mm_stream_load_si128(x); }
	INLINE void xx_store_si128(__m128i * x, __m128i y) { _mm_store_si128(x, y); }
#else
	INLINE __m128i xx_load_si128(const __m128i * x) { return _mm_loadu_si128(x); }
	INLINE void xx_store_si128(__m128i * x, __m128i y) { _mm_storeu_si128(x, y); }
#endif
	INLINE __m128i nop_128(__m128i x) { return x; }

	//

	template <	IntrI op,
				IntrPi load = xx_load_si128,
				IntrPI store = xx_store_si128>
	INLINE void sPtrDst128(const __m128i * pSrc, __m128i * pDst, int len)
	{
#ifdef UNROLL_MORE
		for (; len >= 4; len-=4, pSrc+=4, pDst+=4)
		{
			__m128i a0 = load(pSrc);
			__m128i a1 = load(pSrc+1);
			__m128i a2 = load(pSrc+2);
			__m128i a3 = load(pSrc+3);

			a0 = op(a0);
			a1 = op(a1);
			a2 = op(a2);
			a3 = op(a3);

			store(pDst, a0);
			store(pDst+1, a1);
			store(pDst+2, a2);
			store(pDst+3, a3);
		}
#endif
		for (; len >= 2; len-=2, pSrc+=2, pDst+=2)
		{
			__m128i a0 = load(pSrc);
			__m128i a1 = load(pSrc+1);

			a0 = op(a0);
			a1 = op(a1);

			store(pDst, a0);
			store(pDst+1, a1);
		}

		if (len)
		{
			__m128i a0 = load(pSrc);
			a0 = op(a0);
			store(pDst, a0);
		}
	}

	// TODO
}

namespace sse
{
	using namespace sse_i128_internal;

namespace common
{
	template <IntrPI store_si128 = xx_store_si128>
	INLINE void setT128(__m128i a, __m128i * pDst, int len)
	{
#ifdef UNROLL_MORE
		for (; len >= 8; len-=8, pDst+=8)
		{
			store_si128(pDst, a);
			store_si128(pDst+1, a);
			store_si128(pDst+2, a);
			store_si128(pDst+3, a);

			store_si128(pDst+4, a);
			store_si128(pDst+5, a);
			store_si128(pDst+6, a);
			store_si128(pDst+7, a);
		}
#endif
		for (; len >= 4; len-=4, pDst+=4)
		{
			store_si128(pDst, a);
			store_si128(pDst+1, a);
			store_si128(pDst+2, a);
			store_si128(pDst+3, a);
		}

		if (len >= 2)
		{
			store_si128(pDst, a);
			store_si128(pDst+1, a);
			len -= 2; pDst += 2;
		}

		if (len)
		{
			store_si128(pDst, a);
			--len; --pDst;
		}
	}

	_SIMD_SSE_SPEC void set(int32_t val, int32_t * pDst, int len)
	{
		__m128i a = _mm_set1_epi32(val);
		setT128(a, (__m128i*)pDst, (len>>2));
		for (int i = len - (len & 0x3); i < len; ++i)
			pDst[i] = val;
	}

	_SIMD_SSE_SPEC void set(uint32_t val, uint32_t * pDst, int len)
	{
		set((int32_t)val, (int32_t*)pDst, len);
	}

	_SIMD_SSE_SPEC void set(int64_t val, int64_t * pDst, int len)
	{
		__m128i a = _mm_set1_epi64x(val);
		setT128(a, (__m128i*)pDst, (len>>1));
		if (len)
			pDst[len-1] = val;
	}

	_SIMD_SSE_SPEC void set(uint64_t val, uint64_t * pDst, int len)
	{
		set((int64_t)val, (int64_t*)pDst, len);
	}

	_SIMD_SSE_SPEC void copy(const int32_t * pSrc, int32_t * pDst, int len)
	{
		sPtrDst128<nop_128>((const __m128i*)pSrc, (__m128i*)pDst, (len>>2));
		for (int i = len - (len & 0x3); i < len; ++i)
			pDst[i] = pSrc[i];
	}

	_SIMD_SSE_SPEC void copy(const uint32_t * pSrc, uint32_t * pDst, int len)
	{
		copy((const int32_t*)pSrc, (int32_t*)pDst, len);
	}

	_SIMD_SSE_SPEC void copy(const int64_t * pSrc, int64_t * pDst, int len)
	{
		sPtrDst128<nop_128>((const __m128i*)pSrc, (__m128i*)pDst, (len>>1));
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
	// TODO
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
