#include <stdint.h>
#include <xmmintrin.h>
#include <emmintrin.h>

#include "sse.h"

namespace sse
{
namespace common
{
	template <> void set(float val, float * pDst, int len)
	{
		__m128 a = _mm_set1_ps(val);
#ifdef UNROLL_MORE
		for (; len >= 32; len-=32, pDst+=32)
		{
			_mm_store_ps(pDst, a);
			_mm_store_ps(pDst+4, a);
			_mm_store_ps(pDst+8, a);
			_mm_store_ps(pDst+12, a);

			_mm_store_ps(pDst+16, a);
			_mm_store_ps(pDst+20, a);
			_mm_store_ps(pDst+24, a);
			_mm_store_ps(pDst+28, a);
		}
#endif
		for (; len >= 16; len-=16, pDst+=16)
		{
			_mm_store_ps(pDst, a);
			_mm_store_ps(pDst+4, a);
			_mm_store_ps(pDst+8, a);
			_mm_store_ps(pDst+12, a);
		}

		if (len >= 8)
		{
			_mm_store_ps(pDst, a);
			_mm_store_ps(pDst+4, a);

			len -= 8; pDst += 8;
		}

		if (len >= 4)
		{
			_mm_store_ps(pDst, a);

			len -= 4; pDst += 4;
		}

		for (; len > 0; --len, ++pDst)
			*pDst = val;
	}

	template <> void copy(const float * pSrc, float * pDst, int len)
	{
#ifdef UNROLL_MORE
		for (; len >= 32; len-=32, pSrc+=32, pDst+=32)
		{
			__m128 a0 = _mm_loadu_ps(pSrc);
			__m128 a1 = _mm_loadu_ps(pSrc+4);
			__m128 a2 = _mm_loadu_ps(pSrc+8);
			__m128 a3 = _mm_loadu_ps(pSrc+12);

			__m128 a4 = _mm_loadu_ps(pSrc+16);
			__m128 a5 = _mm_loadu_ps(pSrc+20);
			__m128 a6 = _mm_loadu_ps(pSrc+24);
			__m128 a7 = _mm_loadu_ps(pSrc+28);

			_mm_store_ps(pDst, a0);
			_mm_store_ps(pDst+4, a1);
			_mm_store_ps(pDst+8, a2);
			_mm_store_ps(pDst+12, a3);

			_mm_store_ps(pDst+16, a4);
			_mm_store_ps(pDst+20, a5);
			_mm_store_ps(pDst+24, a6);
			_mm_store_ps(pDst+28, a7);
		}
#endif
		for (; len >= 16; len-=16, pSrc+=16, pDst+=16)
		{
			__m128 a0 = _mm_loadu_ps(pSrc);
			__m128 a1 = _mm_loadu_ps(pSrc+4);
			__m128 a2 = _mm_loadu_ps(pSrc+8);
			__m128 a3 = _mm_loadu_ps(pSrc+12);

			_mm_store_ps(pDst, a0);
			_mm_store_ps(pDst+4, a1);
			_mm_store_ps(pDst+8, a2);
			_mm_store_ps(pDst+12, a3);
		}

		if (len >= 8)
		{
			__m128 a0 = _mm_loadu_ps(pSrc);
			__m128 a1 = _mm_loadu_ps(pSrc+4);

			_mm_store_ps(pDst, a0);
			_mm_store_ps(pDst+4, a1);

			len -= 4; pSrc += 4; pDst += 4;
		}

		if (len >= 4)
		{
			__m128 a0 = _mm_loadu_ps(pSrc);
			_mm_store_ps(pDst, a0);

			len -= 4; pSrc += 4; pDst += 4;
		}

		for (; len > 0; --len, ++pSrc, ++pDst)
			*pDst = *pSrc;
	}

	// double

	template <> void set(double val, double * pDst, int len)
	{
		__m128d a = _mm_set1_pd(val);
#ifdef UNROLL_MORE
		for (; len >= 16; len-=16, pDst+=16)
		{
			_mm_store_pd(pDst, a);
			_mm_store_pd(pDst+2, a);
			_mm_store_pd(pDst+4, a);
			_mm_store_pd(pDst+6, a);

			_mm_store_pd(pDst+8, a);
			_mm_store_pd(pDst+10, a);
			_mm_store_pd(pDst+12, a);
			_mm_store_pd(pDst+14, a);
		}
#endif
		for (; len >= 8; len-=8, pDst+=8)
		{
			_mm_store_pd(pDst, a);
			_mm_store_pd(pDst+2, a);
			_mm_store_pd(pDst+4, a);
			_mm_store_pd(pDst+6, a);
		}

		if (len >= 4)
		{
			_mm_store_pd(pDst, a);
			_mm_store_pd(pDst+2, a);

			len -= 4; pDst += 4;
		}

		if (len >= 2)
		{
			_mm_store_pd(pDst, a);

			len -= 2; pDst += 2;
		}

		if (len)
			*pDst = val;
	}

	template <> void copy(const double * pSrc, double * pDst, int len)
	{
#ifdef UNROLL_MORE
		for (; len >= 16; len-=16, pSrc+=16, pDst+=16)
		{
			__m128d a0 = _mm_loadu_pd(pSrc);
			__m128d a1 = _mm_loadu_pd(pSrc+2);
			__m128d a2 = _mm_loadu_pd(pSrc+4);
			__m128d a3 = _mm_loadu_pd(pSrc+6);

			__m128d a4 = _mm_loadu_pd(pSrc+8);
			__m128d a5 = _mm_loadu_pd(pSrc+10);
			__m128d a6 = _mm_loadu_pd(pSrc+12);
			__m128d a7 = _mm_loadu_pd(pSrc+14);

			_mm_store_pd(pDst, a0);
			_mm_store_pd(pDst+2, a1);
			_mm_store_pd(pDst+4, a2);
			_mm_store_pd(pDst+6, a3);

			_mm_store_pd(pDst+8, a4);
			_mm_store_pd(pDst+10, a5);
			_mm_store_pd(pDst+12, a6);
			_mm_store_pd(pDst+14, a7);
		}
#endif
		for (; len >= 8; len-=8, pSrc+=8, pDst+=8)
		{
			__m128d a0 = _mm_loadu_pd(pSrc);
			__m128d a1 = _mm_loadu_pd(pSrc+2);
			__m128d a2 = _mm_loadu_pd(pSrc+4);
			__m128d a3 = _mm_loadu_pd(pSrc+6);

			_mm_store_pd(pDst, a0);
			_mm_store_pd(pDst+2, a1);
			_mm_store_pd(pDst+4, a2);
			_mm_store_pd(pDst+6, a3);
		}

		if (len >= 4)
		{
			__m128d a0 = _mm_loadu_pd(pSrc);
			__m128d a1 = _mm_loadu_pd(pSrc+2);

			_mm_store_pd(pDst, a0);
			_mm_store_pd(pDst+2, a1);

			len -= 4; pSrc += 4; pDst += 4;
		}

		if (len >= 2)
		{
			__m128d a0 = _mm_loadu_pd(pSrc);
			_mm_store_pd(pDst, a0);

			len -= 2; pSrc += 2; pDst += 2;
		}

		if (len)
			*pDst = *pSrc;
	}
}

namespace arithmetic
{
	template <> void addC(const float * pSrc, float val, float * pDst, int len)
	{
		__m128 b = _mm_set1_ps(val);
#ifdef UNROLL_MORE
		for (; len >= 16; len-=16, pSrc+=16, pDst+=16)
		{
			__m128 a0 = _mm_loadu_ps(pSrc);
			__m128 a1 = _mm_loadu_ps(pSrc+4);
			__m128 a2 = _mm_loadu_ps(pSrc+8);
			__m128 a3 = _mm_loadu_ps(pSrc+12);

			a0 = _mm_add_ps(a0, b);
			a1 = _mm_add_ps(a1, b);
			a2 = _mm_add_ps(a2, b);
			a3 = _mm_add_ps(a3, b);

			_mm_store_ps(pDst, a0);
			_mm_store_ps(pDst+4, a1);
			_mm_store_ps(pDst+8, a2);
			_mm_store_ps(pDst+12, a3);
		}
#endif
		for (; len >= 8; len-=8, pSrc+=8, pDst+=8)
		{
			__m128 a0 = _mm_loadu_ps(pSrc);
			__m128 a1 = _mm_loadu_ps(pSrc+4);

			a0 = _mm_add_ps(a0, b);
			a1 = _mm_add_ps(a1, b);

			_mm_store_ps(pDst, a0);
			_mm_store_ps(pDst+4, a1);
		}

		if (len >= 4)
		{
			__m128 a0 = _mm_loadu_ps(pSrc);

			a0 = _mm_add_ps(a0, b);
			_mm_store_ps(pDst, a0);

			len -= 4; pSrc += 4; pDst += 4;
		}

		for (; len > 0; --len, ++pSrc, ++pDst)
			*pDst = *pSrc + val;
	}

	template <> void add(const float * pSrc1, const float * pSrc2, float * pDst, int len)
	{
#ifdef UNROLL_MORE
		for (; len >= 16; len-=16, pSrc1+=16, pSrc2+=16, pDst+=16)
		{
			__m128 a0 = _mm_loadu_ps(pSrc1);
			__m128 a1 = _mm_loadu_ps(pSrc1+4);
			__m128 a2 = _mm_loadu_ps(pSrc1+8);
			__m128 a3 = _mm_loadu_ps(pSrc1+12);

			__m128 b0 = _mm_loadu_ps(pSrc2);
			__m128 b1 = _mm_loadu_ps(pSrc2+4);
			__m128 b2 = _mm_loadu_ps(pSrc2+8);
			__m128 b3 = _mm_loadu_ps(pSrc2+12);

			a0 = _mm_add_ps(a0, b0);
			a1 = _mm_add_ps(a1, b1);
			a2 = _mm_add_ps(a2, b2);
			a3 = _mm_add_ps(a3, b3);

			_mm_store_ps(pDst, a0);
			_mm_store_ps(pDst+4, a1);
			_mm_store_ps(pDst+8, a2);
			_mm_store_ps(pDst+12, a3);
		}
#endif
		for (; len >= 8; len-=8, pSrc1+=8, pSrc2+=8, pDst+=8)
		{
			__m128 a0 = _mm_loadu_ps(pSrc1);
			__m128 a1 = _mm_loadu_ps(pSrc1+4);

			__m128 b0 = _mm_loadu_ps(pSrc2);
			__m128 b1 = _mm_loadu_ps(pSrc2+4);

			a0 = _mm_add_ps(a0, b0);
			a1 = _mm_add_ps(a1, b1);

			_mm_store_ps(pDst, a0);
			_mm_store_ps(pDst+4, a1);
		}

		if (len >= 4)
		{
			__m128 a0 = _mm_loadu_ps(pSrc1);
			__m128 b0 = _mm_loadu_ps(pSrc2);

			a0 = _mm_add_ps(a0, b0);
			_mm_store_ps(pDst, a0);

			len -= 4; pSrc1 += 4; pSrc2 += 4; pDst += 4;
		}

		for (; len > 0; --len, ++pSrc1, ++pSrc2, ++pDst)
			*pDst = *pSrc1 + *pSrc2;
	}

	template <> void subC(const float * pSrc, float val, float * pDst, int len)
	{
		__m128 b = _mm_set1_ps(val);
#ifdef UNROLL_MORE
		for (; len >= 16; len-=16, pSrc+=16, pDst+=16)
		{
			__m128 a0 = _mm_loadu_ps(pSrc);
			__m128 a1 = _mm_loadu_ps(pSrc+4);
			__m128 a2 = _mm_loadu_ps(pSrc+8);
			__m128 a3 = _mm_loadu_ps(pSrc+12);

			a0 = _mm_sub_ps(a0, b);
			a1 = _mm_sub_ps(a1, b);
			a2 = _mm_sub_ps(a2, b);
			a3 = _mm_sub_ps(a3, b);

			_mm_store_ps(pDst, a0);
			_mm_store_ps(pDst+4, a1);
			_mm_store_ps(pDst+8, a2);
			_mm_store_ps(pDst+12, a3);
		}
#endif
		for (; len >= 8; len-=8, pSrc+=8, pDst+=8)
		{
			__m128 a0 = _mm_loadu_ps(pSrc);
			__m128 a1 = _mm_loadu_ps(pSrc+4);

			a0 = _mm_sub_ps(a0, b);
			a1 = _mm_sub_ps(a1, b);

			_mm_store_ps(pDst, a0);
			_mm_store_ps(pDst+4, a1);
		}

		if (len >= 4)
		{
			__m128 a0 = _mm_loadu_ps(pSrc);
			__m128 r0 = _mm_sub_ps(a0, b);
			_mm_store_ps(pDst, r0);

			len -= 4; pSrc += 4; pDst += 4;
		}

		for (; len > 0; --len, ++pSrc, ++pDst)
			*pDst = *pSrc - val;
	}

	template <> void subCRev(const float * pSrc, float val, float * pDst, int len)
	{
#ifdef UNROLL_MORE
		for (; len >= 16; len-=16, pSrc+=16, pDst+=16)
		{
			__m128 a0 = _mm_set1_ps(val);
			__m128 a1 = _mm_set1_ps(val);
			__m128 a2 = _mm_set1_ps(val);
			__m128 a3 = _mm_set1_ps(val);

			__m128 b0 = _mm_loadu_ps(pSrc);
			__m128 b1 = _mm_loadu_ps(pSrc+4);
			__m128 b2 = _mm_loadu_ps(pSrc+8);
			__m128 b3 = _mm_loadu_ps(pSrc+16);

			a0 = _mm_sub_ps(a0, b0);
			a1 = _mm_sub_ps(a1, b1);
			a2 = _mm_sub_ps(a2, b2);
			a3 = _mm_sub_ps(a3, b3);

			_mm_store_ps(pDst, a0);
			_mm_store_ps(pDst+4, a1);
			_mm_store_ps(pDst+8, a2);
			_mm_store_ps(pDst+12, a3);
		}
#endif
		for (; len >= 8; len-=8, pSrc+=8, pDst+=8)
		{
			__m128 a0 = _mm_set1_ps(val);
			__m128 a1 = _mm_set1_ps(val);

			__m128 b0 = _mm_loadu_ps(pSrc);
			__m128 b1 = _mm_loadu_ps(pSrc+4);

			a0 = _mm_sub_ps(a0, b0);
			a1 = _mm_sub_ps(a1, b1);

			_mm_store_ps(pDst, a0);
			_mm_store_ps(pDst+4, a1);
		}

		if (len >= 4)
		{
			__m128 a0 = _mm_set1_ps(val);
			__m128 b0 = _mm_loadu_ps(pSrc);

			a0 = _mm_sub_ps(a0, b0);
			_mm_store_ps(pDst, a0);

			len -= 4; pSrc += 4; pDst += 4;
		}

		for (; len > 0; --len, ++pSrc, ++pDst)
			*pDst = val - *pSrc;
	}

	template <> void sub(const float * pSrc1, const float * pSrc2, float * pDst, int len)
	{
#ifdef UNROLL_MORE
		for (; len >= 16; len-=16, pSrc1+=16, pSrc2+=16, pDst+=16)
		{
			__m128 a0 = _mm_loadu_ps(pSrc1);
			__m128 a1 = _mm_loadu_ps(pSrc1+4);
			__m128 a2 = _mm_loadu_ps(pSrc1+8);
			__m128 a3 = _mm_loadu_ps(pSrc1+12);

			__m128 b0 = _mm_loadu_ps(pSrc2);
			__m128 b1 = _mm_loadu_ps(pSrc2+4);
			__m128 b2 = _mm_loadu_ps(pSrc2+8);
			__m128 b3 = _mm_loadu_ps(pSrc2+12);

			a0 = _mm_sub_ps(a0, b0);
			a1 = _mm_sub_ps(a1, b1);
			a2 = _mm_sub_ps(a2, b2);
			a3 = _mm_sub_ps(a3, b3);

			_mm_store_ps(pDst, a0);
			_mm_store_ps(pDst+4, a1);
			_mm_store_ps(pDst+8, a2);
			_mm_store_ps(pDst+12, a3);
		}
#endif
		for (; len >= 8; len-=8, pSrc1+=8, pSrc2+=8, pDst+=8)
		{
			__m128 a0 = _mm_loadu_ps(pSrc1);
			__m128 a1 = _mm_loadu_ps(pSrc1+4);

			__m128 b0 = _mm_loadu_ps(pSrc2);
			__m128 b1 = _mm_loadu_ps(pSrc2+4);

			a0 = _mm_sub_ps(a0, b0);
			a1 = _mm_sub_ps(a1, b1);

			_mm_store_ps(pDst, a0);
			_mm_store_ps(pDst+4, a1);
		}

		if (len >= 4)
		{
			__m128 a0 = _mm_loadu_ps(pSrc1);
			__m128 b0 = _mm_loadu_ps(pSrc2);

			a0 = _mm_sub_ps(a0, b0);
			_mm_store_ps(pDst, a0);

			len -= 4; pSrc1 += 4; pSrc2 += 4; pDst += 4;
		}

		for (; len > 0; --len, ++pSrc1, ++pSrc2, ++pDst)
			*pDst = *pSrc1 - *pSrc2;
	}

	template <> void mulC(const float * pSrc, float val, float * pDst, int len)
	{
		__m128 b = _mm_set1_ps(val);
#ifdef UNROLL_MORE
		for (; len >= 16; len-=16, pSrc+=16, pDst+=16)
		{
			__m128 a0 = _mm_loadu_ps(pSrc);
			__m128 a1 = _mm_loadu_ps(pSrc+4);
			__m128 a2 = _mm_loadu_ps(pSrc+8);
			__m128 a3 = _mm_loadu_ps(pSrc+12);

			a0 = _mm_mul_ps(a0, b);
			a1 = _mm_mul_ps(a1, b);
			a2 = _mm_mul_ps(a2, b);
			a3 = _mm_mul_ps(a3, b);

			_mm_store_ps(pDst, a0);
			_mm_store_ps(pDst+4, a1);
			_mm_store_ps(pDst+8, a2);
			_mm_store_ps(pDst+12, a3);
		}
#endif
		for (; len >= 8; len-=8, pSrc+=8, pDst+=8)
		{
			__m128 a0 = _mm_loadu_ps(pSrc);
			__m128 a1 = _mm_loadu_ps(pSrc+4);

			a0 = _mm_mul_ps(a0, b);
			a1 = _mm_mul_ps(a1, b);

			_mm_store_ps(pDst, a0);
			_mm_store_ps(pDst+4, a1);
		}

		if (len >= 4)
		{
			__m128 a0 = _mm_loadu_ps(pSrc);

			a0 = _mm_mul_ps(a0, b);
			_mm_store_ps(pDst, a0);

			len -= 4; pSrc += 4; pDst += 4;
		}

		for (; len > 0; --len, ++pSrc, ++pDst)
			*pDst = *pSrc * val;
	}

	template <> void mul(const float * pSrc1, const float * pSrc2, float * pDst, int len)
	{
#ifdef UNROLL_MORE
		for (; len >= 16; len-=16, pSrc1+=16, pSrc2+=16, pDst+=16)
		{
			__m128 a0 = _mm_loadu_ps(pSrc1);
			__m128 a1 = _mm_loadu_ps(pSrc1+4);
			__m128 a2 = _mm_loadu_ps(pSrc1+8);
			__m128 a3 = _mm_loadu_ps(pSrc1+12);

			__m128 b0 = _mm_loadu_ps(pSrc2);
			__m128 b1 = _mm_loadu_ps(pSrc2+4);
			__m128 b2 = _mm_loadu_ps(pSrc2+8);
			__m128 b3 = _mm_loadu_ps(pSrc2+12);

			a0 = _mm_mul_ps(a0, b0);
			a1 = _mm_mul_ps(a1, b1);
			a2 = _mm_mul_ps(a2, b2);
			a3 = _mm_mul_ps(a3, b3);

			_mm_store_ps(pDst, a0);
			_mm_store_ps(pDst+4, a1);
			_mm_store_ps(pDst+8, a2);
			_mm_store_ps(pDst+12, a3);
		}
#endif
		for (; len >= 8; len-=8, pSrc1+=8, pSrc2+=8, pDst+=8)
		{
			__m128 a0 = _mm_loadu_ps(pSrc1);
			__m128 a1 = _mm_loadu_ps(pSrc1+4);

			__m128 b0 = _mm_loadu_ps(pSrc2);
			__m128 b1 = _mm_loadu_ps(pSrc2+4);

			a0 = _mm_mul_ps(a0, b0);
			a1 = _mm_mul_ps(a1, b1);

			_mm_store_ps(pDst, a0);
			_mm_store_ps(pDst+4, a1);
		}

		if (len >= 4)
		{
			__m128 a0 = _mm_loadu_ps(pSrc1);
			__m128 b0 = _mm_loadu_ps(pSrc2);

			a0 = _mm_mul_ps(a0, b0);
			_mm_store_ps(pDst, a0);

			len -= 4; pSrc1 += 4; pSrc2 += 4; pDst += 4;
		}

		for (; len > 0; --len, ++pSrc1, ++pSrc2, ++pDst)
			*pDst = *pSrc1 * *pSrc2;
	}

	template <> void divC(const float * pSrc, float val, float * pDst, int len)
	{
		__m128 b = _mm_set1_ps(val);
#ifdef UNROLL_MORE
		for (; len >= 16; len-=16, pSrc+=16, pDst+=16)
		{
			__m128 a0 = _mm_loadu_ps(pSrc);
			__m128 a1 = _mm_loadu_ps(pSrc+4);
			__m128 a2 = _mm_loadu_ps(pSrc+8);
			__m128 a3 = _mm_loadu_ps(pSrc+12);

			a0 = _mm_div_ps(a0, b);
			a1 = _mm_div_ps(a1, b);
			a2 = _mm_div_ps(a2, b);
			a3 = _mm_div_ps(a3, b);

			_mm_store_ps(pDst, a0);
			_mm_store_ps(pDst+4, a1);
			_mm_store_ps(pDst+8, a2);
			_mm_store_ps(pDst+12, a3);
		}
#endif
		for (; len >= 8; len-=8, pSrc+=8, pDst+=8)
		{
			__m128 a0 = _mm_loadu_ps(pSrc);
			__m128 a1 = _mm_loadu_ps(pSrc+4);

			a0 = _mm_div_ps(a0, b);
			a1 = _mm_div_ps(a1, b);

			_mm_store_ps(pDst, a0);
			_mm_store_ps(pDst+4, a1);
		}

		if (len >= 4)
		{
			__m128 a0 = _mm_loadu_ps(pSrc);

			a0 = _mm_div_ps(a0, b);
			_mm_store_ps(pDst, a0);

			len -= 4; pSrc += 4; pDst += 4;
		}

		for (; len > 0; --len, ++pSrc, ++pDst)
			*pDst = *pSrc / val;
	}

	template <> void divCRev(const float * pSrc, float val, float * pDst, int len)
	{
#ifdef UNROLL_MORE
		for (; len >= 16; len-=16, pSrc+=16, pDst+=16)
		{
			__m128 a0 = _mm_set1_ps(val);
			__m128 a1 = _mm_set1_ps(val);
			__m128 a2 = _mm_set1_ps(val);
			__m128 a3 = _mm_set1_ps(val);

			__m128 b0 = _mm_loadu_ps(pSrc);
			__m128 b1 = _mm_loadu_ps(pSrc+4);
			__m128 b2 = _mm_loadu_ps(pSrc+8);
			__m128 b3 = _mm_loadu_ps(pSrc+12);

			a0 = _mm_div_ps(a0, b0);
			a1 = _mm_div_ps(a1, b1);
			a2 = _mm_div_ps(a2, b2);
			a3 = _mm_div_ps(a3, b3);

			_mm_store_ps(pDst, a0);
			_mm_store_ps(pDst+4, a1);
			_mm_store_ps(pDst+8, a2);
			_mm_store_ps(pDst+12, a3);
		}
#endif
		for (; len >= 8; len-=8, pSrc+=8, pDst+=8)
		{
			__m128 a0 = _mm_set1_ps(val);
			__m128 a1 = _mm_set1_ps(val);

			__m128 b0 = _mm_loadu_ps(pSrc);
			__m128 b1 = _mm_loadu_ps(pSrc+4);

			a0 = _mm_div_ps(a0, b0);
			a1 = _mm_div_ps(a1, b1);

			_mm_store_ps(pDst, a0);
			_mm_store_ps(pDst+4, a1);
		}

		if (len >= 4)
		{
			__m128 a0 = _mm_set1_ps(val);
			__m128 b0 = _mm_loadu_ps(pSrc);

			a0 = _mm_div_ps(a0, b0);
			_mm_store_ps(pDst, a0);

			len -= 4; pSrc += 4; pDst += 4;
		}

		for (; len > 0; --len, ++pSrc, ++pDst)
			*pDst = val / *pSrc;
	}

	template <> void div(const float * pSrc1, const float * pSrc2, float * pDst, int len)
	{
#ifdef UNROLL_MORE
		for (; len >= 16; len-=16, pSrc1+=16, pSrc2+=16, pDst+=16)
		{
			__m128 a0 = _mm_loadu_ps(pSrc1);
			__m128 a1 = _mm_loadu_ps(pSrc1+4);
			__m128 a2 = _mm_loadu_ps(pSrc1+8);
			__m128 a3 = _mm_loadu_ps(pSrc1+12);

			__m128 b0 = _mm_loadu_ps(pSrc2);
			__m128 b1 = _mm_loadu_ps(pSrc2+4);
			__m128 b2 = _mm_loadu_ps(pSrc2+8);
			__m128 b3 = _mm_loadu_ps(pSrc2+12);

			a0 = _mm_div_ps(a0, b0);
			a1 = _mm_div_ps(a1, b1);
			a2 = _mm_div_ps(a2, b2);
			a3 = _mm_div_ps(a3, b3);

			_mm_store_ps(pDst, a0);
			_mm_store_ps(pDst+4, a1);
			_mm_store_ps(pDst+8, a2);
			_mm_store_ps(pDst+12, a3);
		}
#endif
		for (; len >= 8; len-=8, pSrc1+=8, pSrc2+=8, pDst+=8)
		{
			__m128 a0 = _mm_loadu_ps(pSrc1);
			__m128 a1 = _mm_loadu_ps(pSrc1+4);

			__m128 b0 = _mm_loadu_ps(pSrc2);
			__m128 b1 = _mm_loadu_ps(pSrc2+4);

			a0 = _mm_div_ps(a0, b0);
			a1 = _mm_div_ps(a1, b1);

			_mm_store_ps(pDst, a0);
			_mm_store_ps(pDst+4, a1);
		}

		if (len >= 4)
		{
			__m128 a0 = _mm_loadu_ps(pSrc1);
			__m128 b0 = _mm_loadu_ps(pSrc2);

			a0 = _mm_div_ps(a0, b0);
			_mm_store_ps(pDst, a0);

			len -= 4; pSrc1 += 4; pSrc2 += 4; pDst += 4;
		}

		for (; len > 0; --len, ++pSrc1, ++pSrc2, ++pDst)
			*pDst = *pSrc1 / *pSrc2;
	}

	// double

	template <> void addC(const double * pSrc, double val, double * pDst, int len)
	{
		__m128d b = _mm_set1_pd(val);
#ifdef UNROLL_MORE
		for (; len >= 8; len-=8, pSrc+=8, pDst+=8)
		{
			__m128d a0 = _mm_loadu_pd(pSrc);
			__m128d a1 = _mm_loadu_pd(pSrc+2);
			__m128d a2 = _mm_loadu_pd(pSrc+4);
			__m128d a3 = _mm_loadu_pd(pSrc+6);

			a0 = _mm_add_pd(a0, b);
			a1 = _mm_add_pd(a1, b);
			a2 = _mm_add_pd(a2, b);
			a3 = _mm_add_pd(a3, b);

			_mm_store_pd(pDst, a0);
			_mm_store_pd(pDst+2, a1);
			_mm_store_pd(pDst+4, a2);
			_mm_store_pd(pDst+6, a3);
		}
#endif
		for (; len >= 4; len-=4, pSrc+=4, pDst+=4)
		{
			__m128d a0 = _mm_loadu_pd(pSrc);
			__m128d a1 = _mm_loadu_pd(pSrc+2);

			a0 = _mm_add_pd(a0, b);
			a1 = _mm_add_pd(a1, b);

			_mm_store_pd(pDst, a0);
			_mm_store_pd(pDst+2, a1);
		}

		if (len >= 2)
		{
			__m128d a0 = _mm_loadu_pd(pSrc);
			a0 = _mm_add_pd(a0, b);
			_mm_store_pd(pDst, a0);

			len -= 2; pSrc += 2; pDst += 2;
 		}

		for (; len > 0; --len, ++pSrc, ++pDst)
			*pDst = *pSrc + val;
	}

	template <> void add(const double * pSrc1, const double * pSrc2, double * pDst, int len)
	{
#ifdef UNROLL_MORE
		for (; len >= 8; len-=8, pSrc1+=8, pSrc2+=8, pDst+=8)
		{
			__m128d a0 = _mm_loadu_pd(pSrc1);
			__m128d a1 = _mm_loadu_pd(pSrc1+2);
			__m128d a2 = _mm_loadu_pd(pSrc1+4);
			__m128d a3 = _mm_loadu_pd(pSrc1+6);

			__m128d b0 = _mm_loadu_pd(pSrc2);
			__m128d b1 = _mm_loadu_pd(pSrc2+2);
			__m128d b2 = _mm_loadu_pd(pSrc2+4);
			__m128d b3 = _mm_loadu_pd(pSrc2+6);

			a0 = _mm_add_pd(a0, b0);
			a1 = _mm_add_pd(a1, b1);
			a2 = _mm_add_pd(a2, b2);
			a3 = _mm_add_pd(a3, b3);

			_mm_store_pd(pDst, a0);
			_mm_store_pd(pDst+2, a1);
			_mm_store_pd(pDst+4, a2);
			_mm_store_pd(pDst+6, a3);
		}
#endif
		for (; len >= 4; len-=4, pSrc1+=4, pSrc2+=4, pDst+=4)
		{
			__m128d a0 = _mm_loadu_pd(pSrc1);
			__m128d a1 = _mm_loadu_pd(pSrc1+2);

			__m128d b0 = _mm_loadu_pd(pSrc2);
			__m128d b1 = _mm_loadu_pd(pSrc2+2);

			a0 = _mm_add_pd(a0, b0);
			a1 = _mm_add_pd(a1, b1);

			_mm_store_pd(pDst, a0);
			_mm_store_pd(pDst+2, a1);
		}

		if (len >= 2)
		{
			__m128d a0 = _mm_loadu_pd(pSrc1);
			__m128d b0 = _mm_loadu_pd(pSrc2);

			a0 = _mm_add_pd(a0, b0);
			_mm_store_pd(pDst, a0);

			len -= 2; pSrc1 += 2; pSrc2 += 2; pDst += 2;
 		}

		for (; len > 0; --len, ++pSrc1, ++pSrc2, ++pDst)
			*pDst = *pSrc1 + *pSrc2;
	}

	template <> void subC(const double * pSrc, double val, double * pDst, int len)
	{
		__m128d b = _mm_set1_pd(val);
#ifdef UNROLL_MORE
		for (; len >= 8; len-=8, pSrc+=8, pDst+=8)
		{
			__m128d a0 = _mm_loadu_pd(pSrc);
			__m128d a1 = _mm_loadu_pd(pSrc+2);
			__m128d a2 = _mm_loadu_pd(pSrc+4);
			__m128d a3 = _mm_loadu_pd(pSrc+6);

			a0 = _mm_sub_pd(a0, b);
			a1 = _mm_sub_pd(a1, b);
			a2 = _mm_sub_pd(a2, b);
			a3 = _mm_sub_pd(a3, b);

			_mm_store_pd(pDst, a0);
			_mm_store_pd(pDst+2, a1);
			_mm_store_pd(pDst+4, a2);
			_mm_store_pd(pDst+6, a3);
		}
#endif
		for (; len >= 4; len-=4, pSrc+=4, pDst+=4)
		{
			__m128d a0 = _mm_loadu_pd(pSrc);
			__m128d a1 = _mm_loadu_pd(pSrc+2);

			a0 = _mm_sub_pd(a0, b);
			a1 = _mm_sub_pd(a1, b);

			_mm_store_pd(pDst, a0);
			_mm_store_pd(pDst+2, a1);
		}

		if (len >= 2)
		{
			__m128d a0 = _mm_loadu_pd(pSrc);
			a0 = _mm_sub_pd(a0, b);
			_mm_store_pd(pDst, a0);

			len -= 2; pSrc += 2; pDst += 2;
		}

		for (; len > 0; --len, ++pSrc, ++pDst)
			*pDst = *pSrc - val;
	}

	template <> void subCRev(const double * pSrc, double val, double * pDst, int len)
	{
#ifdef UNROLL_MORE
		for (; len >= 8; len-=8, pSrc+=8, pDst+=8)
		{
			__m128d a0 = _mm_set1_pd(val);
			__m128d a1 = _mm_set1_pd(val);
			__m128d a2 = _mm_set1_pd(val);
			__m128d a3 = _mm_set1_pd(val);

			__m128d b0 = _mm_loadu_pd(pSrc);
			__m128d b1 = _mm_loadu_pd(pSrc+2);
			__m128d b2 = _mm_loadu_pd(pSrc+4);
			__m128d b3 = _mm_loadu_pd(pSrc+6);

			a0 = _mm_sub_pd(a0, b0);
			a1 = _mm_sub_pd(a1, b1);
			a2 = _mm_sub_pd(a2, b2);
			a3 = _mm_sub_pd(a3, b3);

			_mm_store_pd(pDst, a0);
			_mm_store_pd(pDst+2, a1);
			_mm_store_pd(pDst+4, a2);
			_mm_store_pd(pDst+6, a3);
		}
#endif
		for (; len >= 4; len-=4, pSrc+=4, pDst+=4)
		{
			__m128d a0 = _mm_set1_pd(val);
			__m128d a1 = _mm_set1_pd(val);

			__m128d b0 = _mm_loadu_pd(pSrc);
			__m128d b1 = _mm_loadu_pd(pSrc+2);

			a0 = _mm_sub_pd(a0, b0);
			a1 = _mm_sub_pd(a1, b1);

			_mm_store_pd(pDst, a0);
			_mm_store_pd(pDst+2, a1);
		}

		if (len >= 2)
		{
			__m128d a0 = _mm_set1_pd(val);
			__m128d b0 = _mm_loadu_pd(pSrc);

			a0 = _mm_sub_pd(a0, b0);
			_mm_store_pd(pDst, a0);

			len -= 2; pSrc += 2; pDst += 2;
		}

		for (; len > 0; --len, ++pSrc, ++pDst)
			*pDst = val - *pSrc;
	}

	template <> void sub(const double * pSrc1, const double * pSrc2, double * pDst, int len)
	{
#ifdef UNROLL_MORE
		for (; len >= 8; len-=8, pSrc1+=8, pSrc2+=8, pDst+=8)
		{
			__m128d a0 = _mm_loadu_pd(pSrc1);
			__m128d a1 = _mm_loadu_pd(pSrc1+2);
			__m128d a2 = _mm_loadu_pd(pSrc1+4);
			__m128d a3 = _mm_loadu_pd(pSrc1+6);

			__m128d b0 = _mm_loadu_pd(pSrc2);
			__m128d b1 = _mm_loadu_pd(pSrc2+2);
			__m128d b2 = _mm_loadu_pd(pSrc2+4);
			__m128d b3 = _mm_loadu_pd(pSrc2+6);

			a0 = _mm_sub_pd(a0, b0);
			a1 = _mm_sub_pd(a1, b1);
			a2 = _mm_sub_pd(a2, b2);
			a3 = _mm_sub_pd(a3, b3);

			_mm_store_pd(pDst, a0);
			_mm_store_pd(pDst+2, a1);
			_mm_store_pd(pDst+4, a2);
			_mm_store_pd(pDst+6, a3);
		}
#endif
		for (; len >= 4; len-=4, pSrc1+=4, pSrc2+=4, pDst+=4)
		{
			__m128d a0 = _mm_loadu_pd(pSrc1);
			__m128d a1 = _mm_loadu_pd(pSrc1+2);

			__m128d b0 = _mm_loadu_pd(pSrc2);
			__m128d b1 = _mm_loadu_pd(pSrc2+2);

			a0 = _mm_sub_pd(a0, b0);
			a1 = _mm_sub_pd(a1, b1);

			_mm_store_pd(pDst, a0);
			_mm_store_pd(pDst+2, a1);
		}

		if (len >= 2)
		{
			__m128d a0 = _mm_loadu_pd(pSrc1);
			__m128d b0 = _mm_loadu_pd(pSrc2);

			a0 = _mm_sub_pd(a0, b0);
			_mm_store_pd(pDst, a0);

			len -= 2; pSrc1 += 2; pSrc2 += 2; pDst += 2;
		}

		for (; len > 0; --len, ++pSrc1, ++pSrc2, ++pDst)
			*pDst = *pSrc1 - *pSrc2;
	}

	template <> void mulC(const double * pSrc, double val, double * pDst, int len)
	{
		__m128d b = _mm_set1_pd(val);
#ifdef UNROLL_MORE
		for (; len >= 8; len-=8, pSrc+=8, pDst+=8)
		{
			__m128d a0 = _mm_loadu_pd(pSrc);
			__m128d a1 = _mm_loadu_pd(pSrc+2);
			__m128d a2 = _mm_loadu_pd(pSrc+4);
			__m128d a3 = _mm_loadu_pd(pSrc+6);

			a0 = _mm_mul_pd(a0, b);
			a1 = _mm_mul_pd(a1, b);
			a2 = _mm_mul_pd(a2, b);
			a3 = _mm_mul_pd(a3, b);

			_mm_store_pd(pDst, a0);
			_mm_store_pd(pDst+2, a1);
			_mm_store_pd(pDst+4, a2);
			_mm_store_pd(pDst+6, a3);
		}
#endif
		for (; len >= 4; len-=4, pSrc+=4, pDst+=4)
		{
			__m128d a0 = _mm_loadu_pd(pSrc);
			__m128d a1 = _mm_loadu_pd(pSrc+2);

			a0 = _mm_mul_pd(a0, b);
			a1 = _mm_mul_pd(a1, b);

			_mm_store_pd(pDst, a0);
			_mm_store_pd(pDst+2, a1);
		}

		if (len >= 2)
		{
			__m128d a0 = _mm_loadu_pd(pSrc);

			a0 = _mm_mul_pd(a0, b);
			_mm_store_pd(pDst, a0);

			len -= 2; pSrc += 2; pDst += 2;
		}

		for (; len > 0; --len, ++pSrc, ++pDst)
			*pDst = *pSrc * val;
	}

	template <> void mul(const double * pSrc1, const double * pSrc2, double * pDst, int len)
	{
#ifdef UNROLL_MORE
		for (; len >= 8; len-=8, pSrc1+=8, pSrc2+=8, pDst+=8)
		{
			__m128d a0 = _mm_loadu_pd(pSrc1);
			__m128d a1 = _mm_loadu_pd(pSrc1+2);
			__m128d a2 = _mm_loadu_pd(pSrc1+4);
			__m128d a3 = _mm_loadu_pd(pSrc1+6);

			__m128d b0 = _mm_loadu_pd(pSrc2);
			__m128d b1 = _mm_loadu_pd(pSrc2+2);
			__m128d b2 = _mm_loadu_pd(pSrc2+4);
			__m128d b3 = _mm_loadu_pd(pSrc2+6);

			a0 = _mm_mul_pd(a0, b0);
			a1 = _mm_mul_pd(a1, b1);
			a2 = _mm_mul_pd(a2, b2);
			a3 = _mm_mul_pd(a3, b3);

			_mm_store_pd(pDst, a0);
			_mm_store_pd(pDst+2, a1);
			_mm_store_pd(pDst+4, a2);
			_mm_store_pd(pDst+6, a3);
		}
#endif
		for (; len >= 4; len-=4, pSrc1+=4, pSrc2+=4, pDst+=4)
		{
			__m128d a0 = _mm_loadu_pd(pSrc1);
			__m128d a1 = _mm_loadu_pd(pSrc1+2);

			__m128d b0 = _mm_loadu_pd(pSrc2);
			__m128d b1 = _mm_loadu_pd(pSrc2+2);

			a0 = _mm_mul_pd(a0, b0);
			a1 = _mm_mul_pd(a1, b1);

			_mm_store_pd(pDst, a0);
			_mm_store_pd(pDst+2, a1);
		}

		if (len >= 2)
		{
			__m128d a0 = _mm_loadu_pd(pSrc1);
			__m128d b0 = _mm_loadu_pd(pSrc2);

			a0 = _mm_mul_pd(a0, b0);
			_mm_store_pd(pDst, a0);

			len -= 2; pSrc1 += 2; pSrc2 += 2; pDst += 2;
		}

		for (; len > 0; --len, ++pSrc1, ++pSrc2, ++pDst)
			*pDst = *pSrc1 * *pSrc2;
	}

	template <> void divC(const double * pSrc, double val, double * pDst, int len)
	{
		__m128d b = _mm_set1_pd(val);
#ifdef UNROLL_MORE
		for (; len >= 8; len-=8, pSrc+=8, pDst+=8)
		{
			__m128d a0 = _mm_loadu_pd(pSrc);
			__m128d a1 = _mm_loadu_pd(pSrc+2);
			__m128d a2 = _mm_loadu_pd(pSrc+4);
			__m128d a3 = _mm_loadu_pd(pSrc+6);

			a0 = _mm_div_pd(a0, b);
			a1 = _mm_div_pd(a1, b);
			a2 = _mm_div_pd(a2, b);
			a3 = _mm_div_pd(a3, b);

			_mm_store_pd(pDst, a0);
			_mm_store_pd(pDst+2, a1);
			_mm_store_pd(pDst+4, a2);
			_mm_store_pd(pDst+6, a3);
		}
#endif
		for (; len >= 4; len-=4, pSrc+=4, pDst+=4)
		{
			__m128d a0 = _mm_loadu_pd(pSrc);
			__m128d a1 = _mm_loadu_pd(pSrc+2);

			a0 = _mm_div_pd(a0, b);
			a1 = _mm_div_pd(a1, b);

			_mm_store_pd(pDst, a0);
			_mm_store_pd(pDst+2, a1);
		}

		if (len >= 2)
		{
			__m128d a0 = _mm_loadu_pd(pSrc);

			a0 = _mm_div_pd(a0, b);
			_mm_store_pd(pDst, a0);

			len -= 2; pSrc += 2; pDst += 2;
		}

		for (; len > 0; --len, ++pSrc, ++pDst)
			*pDst = *pSrc / val;
	}

	template <> void divCRev(const double * pSrc, double val, double * pDst, int len)
	{
#ifdef UNROLL_MORE
		for (; len >= 8; len-=8, pSrc+=8, pDst+=8)
		{
			__m128d a0 = _mm_set1_pd(val);
			__m128d a1 = _mm_set1_pd(val);
			__m128d a2 = _mm_set1_pd(val);
			__m128d a3 = _mm_set1_pd(val);

			__m128d b0 = _mm_loadu_pd(pSrc);
			__m128d b1 = _mm_loadu_pd(pSrc+2);
			__m128d b2 = _mm_loadu_pd(pSrc+4);
			__m128d b3 = _mm_loadu_pd(pSrc+6);

			a0 = _mm_div_pd(a0, b0);
			a1 = _mm_div_pd(a1, b1);
			a2 = _mm_div_pd(a2, b2);
			a3 = _mm_div_pd(a3, b3);

			_mm_store_pd(pDst, a0);
			_mm_store_pd(pDst+2, a1);
			_mm_store_pd(pDst+4, a2);
			_mm_store_pd(pDst+6, a3);
		}
#endif
		for (; len >= 4; len-=4, pSrc+=4, pDst+=4)
		{
			__m128d a0 = _mm_set1_pd(val);
			__m128d a1 = _mm_set1_pd(val);

			__m128d b0 = _mm_loadu_pd(pSrc);
			__m128d b1 = _mm_loadu_pd(pSrc+2);

			a0 = _mm_div_pd(a0, b0);
			a1 = _mm_div_pd(a1, b1);

			_mm_store_pd(pDst, a0);
			_mm_store_pd(pDst+2, a1);
		}

		if (len >= 2)
		{
			__m128d a0 = _mm_set1_pd(val);
			__m128d b0 = _mm_loadu_pd(pSrc);

			a0 = _mm_div_pd(a0, b0);
			_mm_store_pd(pDst, a0);

			len -= 2; pSrc += 2; pDst += 2;
		}

		for (; len > 0; --len, ++pSrc, ++pDst)
			*pDst = val / *pSrc;
	}

	template <> void div(const double * pSrc1, const double * pSrc2, double * pDst, int len)
	{
#ifdef UNROLL_MORE
		for (; len >= 8; len-=8, pSrc1+=8, pSrc2+=8, pDst+=8)
		{
			__m128d a0 = _mm_loadu_pd(pSrc1);
			__m128d a1 = _mm_loadu_pd(pSrc1+2);
			__m128d a2 = _mm_loadu_pd(pSrc1+4);
			__m128d a3 = _mm_loadu_pd(pSrc1+6);

			__m128d b0 = _mm_loadu_pd(pSrc2);
			__m128d b1 = _mm_loadu_pd(pSrc2+2);
			__m128d b2 = _mm_loadu_pd(pSrc2+4);
			__m128d b3 = _mm_loadu_pd(pSrc2+6);

			a0 = _mm_div_pd(a0, b0);
			a1 = _mm_div_pd(a1, b1);
			a2 = _mm_div_pd(a2, b2);
			a3 = _mm_div_pd(a3, b3);

			_mm_store_pd(pDst, a0);
			_mm_store_pd(pDst+2, a1);
			_mm_store_pd(pDst+4, a2);
			_mm_store_pd(pDst+6, a3);
		}
#endif
		for (; len >= 4; len-=4, pSrc1+=4, pSrc2+=4, pDst+=4)
		{
			__m128d a0 = _mm_loadu_pd(pSrc1);
			__m128d a1 = _mm_loadu_pd(pSrc1+2);

			__m128d b0 = _mm_loadu_pd(pSrc2);
			__m128d b1 = _mm_loadu_pd(pSrc2+2);

			a0 = _mm_div_pd(a0, b0);
			a1 = _mm_div_pd(a1, b1);

			_mm_store_pd(pDst, a0);
			_mm_store_pd(pDst+2, a1);
		}

		if (len >= 2)
		{
			__m128d a0 = _mm_loadu_pd(pSrc1);
			__m128d b0 = _mm_loadu_pd(pSrc2);

			a0 = _mm_div_pd(a0, b0);
			_mm_store_pd(pDst, a0);

			len -= 2; pSrc1 += 2; pSrc2 += 2; pDst += 2;
		}

		for (; len > 0; --len, ++pSrc1, ++pSrc2, ++pDst)
			*pDst = *pSrc1 / *pSrc2;
	}
}

namespace power
{
	template <> void inv(const float * pSrc, float * pDst, int len)
	{
		for (; len >= 16; len-=16, pSrc+=16, pDst+=16)
		{
			__m128 a0 = _mm_loadu_ps(pSrc);
			__m128 a1 = _mm_loadu_ps(pSrc+4);
			__m128 a2 = _mm_loadu_ps(pSrc+8);
			__m128 a3 = _mm_loadu_ps(pSrc+12);

			_mm_store_ps(pDst, 		_mm_rcp_ps(a0));
			_mm_store_ps(pDst+4,	_mm_rcp_ps(a1));
			_mm_store_ps(pDst+8,	_mm_rcp_ps(a2));
			_mm_store_ps(pDst+12,	_mm_rcp_ps(a3));
		}

		if (len >= 8)
		{
			__m128 a0 = _mm_loadu_ps(pSrc);
			__m128 a1 = _mm_loadu_ps(pSrc+4);

			_mm_store_ps(pDst, 		_mm_rcp_ps(a0));
			_mm_store_ps(pDst+4,	_mm_rcp_ps(a1));

			len -= 8; pSrc += 8; pDst += 8;
		}

		if (len >= 4)
		{
			__m128 a0 = _mm_loadu_ps(pSrc);
			_mm_store_ps(pDst,		_mm_rcp_ps(a0));

			len -= 4; pSrc += 4; pDst += 4;
		}

		for (; len > 0; --len, ++pSrc, ++pDst)
		{
			__m128 a0 = _mm_load_ss(pSrc);
			_mm_store_ss(pDst,		_mm_rcp_ss(a0));
		}
	}

	template <> void sqrt(const float * pSrc, float * pDst, int len)
	{
		for (; len >= 16; len-=16, pSrc+=16, pDst+=16)
		{
			__m128 a0 = _mm_loadu_ps(pSrc);
			__m128 a1 = _mm_loadu_ps(pSrc+4);
			__m128 a2 = _mm_loadu_ps(pSrc+8);
			__m128 a3 = _mm_loadu_ps(pSrc+12);

			_mm_store_ps(pDst, 		_mm_sqrt_ps(a0));
			_mm_store_ps(pDst+4,	_mm_sqrt_ps(a1));
			_mm_store_ps(pDst+8,	_mm_sqrt_ps(a2));
			_mm_store_ps(pDst+12,	_mm_sqrt_ps(a3));
		}

		if (len >= 8)
		{
			__m128 a0 = _mm_loadu_ps(pSrc);
			__m128 a1 = _mm_loadu_ps(pSrc+4);

			_mm_store_ps(pDst,		_mm_sqrt_ps(a0));
			_mm_store_ps(pDst+4,	_mm_sqrt_ps(a1));

			len -= 8; pSrc += 8; pDst += 8;
		}

		if (len >= 4)
		{
			__m128 a0 = _mm_loadu_ps(pSrc);
			_mm_store_ps(pDst,		_mm_sqrt_ps(a0));

			len -= 4, pSrc += 4; pDst += 4;
		}

		for (; len > 0; --len, ++pSrc, ++pDst)
		{
			__m128 a0 = _mm_load_ss(pSrc);
			_mm_store_ss(pDst,		_mm_sqrt_ss(a0));
		}
	}

	template <> void invSqrt(const float * pSrc, float * pDst, int len)
	{
		for (; len >= 16; len-=16, pSrc+=16, pDst+=16)
		{
			__m128 a0 = _mm_loadu_ps(pSrc);
			__m128 a1 = _mm_loadu_ps(pSrc+4);
			__m128 a2 = _mm_loadu_ps(pSrc+8);
			__m128 a3 = _mm_loadu_ps(pSrc+12);

			_mm_store_ps(pDst,		_mm_rsqrt_ps(a0));
			_mm_store_ps(pDst+4,	_mm_rsqrt_ps(a1));
			_mm_store_ps(pDst+8,	_mm_rsqrt_ps(a2));
			_mm_store_ps(pDst+12,	_mm_rsqrt_ps(a3));
		}

		if (len >= 8)
		{
			__m128 a0 = _mm_loadu_ps(pSrc);
			__m128 a1 = _mm_loadu_ps(pSrc+4);

			_mm_store_ps(pDst,		_mm_rsqrt_ps(a0));
			_mm_store_ps(pDst+4,	_mm_rsqrt_ps(a1));

			len -= 8; pSrc += 8; pDst += 8;
		}

		if (len >= 4)
		{
			__m128 a0 = _mm_loadu_ps(pSrc);
			_mm_store_ps(pDst, 		_mm_rsqrt_ps(a0));

			len -= 4; pSrc += 4; pDst += 4;
		}

		for (; len > 0; --len, ++pSrc, ++pDst)
		{
			__m128 a0 = _mm_load_ss(pSrc);
			_mm_store_ss(pDst,		_mm_rsqrt_ss(a0));
		}
	}

	// double

	template <> void inv(const double * pSrc, double * pDst, int len)
	{
		divCRev(pSrc, 1.0, pDst, len);
	}

	template <> void sqrt(const double * pSrc, double * pDst, int len)
	{
		for (; len >= 8; len-=8, pSrc+=8, pDst+=8)
		{
			__m128d a0 = _mm_loadu_pd(pSrc);
			__m128d a1 = _mm_loadu_pd(pSrc+2);
			__m128d a2 = _mm_loadu_pd(pSrc+4);
			__m128d a3 = _mm_loadu_pd(pSrc+6);

			_mm_store_pd(pDst,		_mm_sqrt_pd(a0));
			_mm_store_pd(pDst+2,	_mm_sqrt_pd(a1));
			_mm_store_pd(pDst+4,	_mm_sqrt_pd(a2));
			_mm_store_pd(pDst+6,	_mm_sqrt_pd(a3));
		}

		if (len >= 4)
		{
			__m128d a0 = _mm_loadu_pd(pSrc);
			__m128d a1 = _mm_loadu_pd(pSrc+2);

			_mm_store_pd(pDst,		_mm_sqrt_pd(a0));
			_mm_store_pd(pDst+2,	_mm_sqrt_pd(a1));

			len -= 4; pSrc += 4; pDst += 4;
		}

		if (len >= 2)
		{
			__m128d a0 = _mm_loadu_pd(pSrc);
			_mm_store_pd(pDst,		_mm_sqrt_pd(a0));

			len -= 2; pSrc += 2; pDst += 2;
		}

		if (len)
		{
			__m128d a0 = _mm_load_sd(pSrc);
			__m128d r0;
			_mm_sqrt_sd(a0, r0);
			_mm_store_sd(pDst, r0);
		}
	}

	template <> void invSqrt(const double * pSrc, double * pDst, int len)
	{
		sqrt(pSrc, pDst, len);
		inv(pDst, pDst, len);
	}
}

namespace statistical
{
	template <> void min(const float * pSrc, int len, float * pMin)
	{
		if (len < 16) {
			nosimd::min(pSrc, len, pMin);
			return;
		}

		__m128 m0 = _mm_loadu_ps(pSrc);
		__m128 m1 = _mm_loadu_ps(pSrc+4);
		__m128 m2 = _mm_loadu_ps(pSrc+8);
		__m128 m3 = _mm_loadu_ps(pSrc+12);

		len -= 16;
		pSrc += 16;

		for (; len >= 16; len-=16, pSrc+=16)
		{
			__m128 a0 = _mm_loadu_ps(pSrc);
			__m128 a1 = _mm_loadu_ps(pSrc+4);
			__m128 a2 = _mm_loadu_ps(pSrc+8);
			__m128 a3 = _mm_loadu_ps(pSrc+12);

			m0 = _mm_min_ps(m0, a0);
			m1 = _mm_min_ps(m1, a1);
			m2 = _mm_min_ps(m2, a2);
			m3 = _mm_min_ps(m3, a3);
		}

		m0 = _mm_min_ps(m0, m1);
		m2 = _mm_min_ps(m2, m3);
		m0 = _mm_min_ps(m0, m2);

		for (; len >= 4; len-=4, pSrc+=4)
		{
			__m128 a0 = _mm_loadu_ps(pSrc);
			m0 = _mm_min_ps(m0, a0);
		}

		float res[4];
		_mm_store_ps(res, m0);

		*pMin = res[0];
		for (unsigned i=1; i<4; ++i)
			if (*pMin < res[i])
				*pMin = res[i];

		for (; len > 0; --len, ++pSrc)
			if (*pMin < *pSrc)
				*pMin = *pSrc;
	}
}
}

