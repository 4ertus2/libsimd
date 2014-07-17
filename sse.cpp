#include <stdint.h>
#include <xmmintrin.h>
#include <emmintrin.h>

#include "sse.h"

namespace sse
{
namespace arithmetic
{
	template <> void addC(const float * pSrc, float val, float * pDst, int len)
	{
		__m128 b = _mm_load1_ps(&val);
	
		for (; len >= 8; len-=8, pSrc+=8, pDst+=8)
		{
			__m128 a0 = _mm_loadu_ps(pSrc);
			__m128 a1 = _mm_loadu_ps(pSrc+4);
			
			__m128 r0 = _mm_add_ps(a0, b);
			__m128 r1 = _mm_add_ps(a1, b);
			
			_mm_store_ps(pDst, r0);
			_mm_store_ps(pDst+4, r1);
		}
	
		if (len >= 4)
		{
			__m128 a0 = _mm_loadu_ps(pSrc);
			__m128 r0 = _mm_add_ps(a0, b);
			_mm_store_ps(pDst, r0);
			
			len -= 4;
			pSrc += 4;
			pDst += 4;
		}

		for (; len > 0; --len, ++pSrc, ++pDst)
			*pDst = *pSrc + val;
	}

	template <> void add(const float * pSrc1, const float * pSrc2, float * pDst, int len)
	{
		for (; len >= 8; len-=8, pSrc1+=8, pSrc2+=8, pDst+=8)
		{
			__m128 a0 = _mm_loadu_ps(pSrc1);
			__m128 a1 = _mm_loadu_ps(pSrc1+4);
			__m128 b0 = _mm_loadu_ps(pSrc2);
			__m128 b1 = _mm_loadu_ps(pSrc2+4);
			
			__m128 r0 = _mm_add_ps(a0, b0);
			__m128 r1 = _mm_add_ps(a1, b1);
			
			_mm_store_ps(pDst, r0);
			_mm_store_ps(pDst+4, r1);
		}
	
		if (len >= 4)
		{
			__m128 a0 = _mm_loadu_ps(pSrc1);
			__m128 b0 = _mm_loadu_ps(pSrc2);

			__m128 r0 = _mm_add_ps(a0, b0);
			_mm_store_ps(pDst, r0);
			
			len -= 4;
			pSrc1 += 4;
			pSrc2 += 4;
			pDst += 4;
		}

		for (; len > 0; --len, ++pSrc1, ++pSrc2, ++pDst)
			*pDst = *pSrc1 + *pSrc2;
	}

	template <> void subC(const float * pSrc, float val, float * pDst, int len)
	{
		__m128 b = _mm_load1_ps(&val);
	
		for (; len >= 8; len-=8, pSrc+=8, pDst+=8)
		{
			__m128 a0 = _mm_loadu_ps(pSrc);
			__m128 a1 = _mm_loadu_ps(pSrc+4);
			
			__m128 r0 = _mm_sub_ps(a0, b);
			__m128 r1 = _mm_sub_ps(a1, b);
			
			_mm_store_ps(pDst, r0);
			_mm_store_ps(pDst+4, r1);
		}
	
		if (len >= 4)
		{
			__m128 a0 = _mm_loadu_ps(pSrc);
			__m128 r0 = _mm_sub_ps(a0, b);
			_mm_store_ps(pDst, r0);
			
			len -= 4;
			pSrc += 4;
			pDst += 4;
		}

		for (; len > 0; --len, ++pSrc, ++pDst)
			*pDst = *pSrc - val;
	}

	template <> void subCRev(const float * pSrc, float val, float * pDst, int len)
	{
		__m128 a = _mm_load1_ps(&val);
	
		for (; len >= 8; len-=8, pSrc+=8, pDst+=8)
		{
			__m128 b0 = _mm_loadu_ps(pSrc);
			__m128 b1 = _mm_loadu_ps(pSrc+4);
			
			__m128 r0 = _mm_sub_ps(a, b0);
			__m128 r1 = _mm_sub_ps(a, b1);
			
			_mm_store_ps(pDst, r0);
			_mm_store_ps(pDst+4, r1);
		}
	
		if (len >= 4)
		{
			__m128 b0 = _mm_loadu_ps(pSrc);
			__m128 r0 = _mm_sub_ps(a, b0);
			_mm_store_ps(pDst, r0);
			
			len -= 4;
			pSrc += 4;
			pDst += 4;
		}

		for (; len > 0; --len, ++pSrc, ++pDst)
			*pDst = val - *pSrc;
	}

	template <> void sub(const float * pSrc1, const float * pSrc2, float * pDst, int len)
	{
		for (; len >= 8; len-=8, pSrc1+=8, pSrc2+=8, pDst+=8)
		{
			__m128 a0 = _mm_loadu_ps(pSrc1);
			__m128 a1 = _mm_loadu_ps(pSrc1+4);
			__m128 b0 = _mm_loadu_ps(pSrc2);
			__m128 b1 = _mm_loadu_ps(pSrc2+4);
			
			__m128 r0 = _mm_sub_ps(a0, b0);
			__m128 r1 = _mm_sub_ps(a1, b1);
			
			_mm_store_ps(pDst, r0);
			_mm_store_ps(pDst+4, r1);
		}
	
		if (len >= 4)
		{
			__m128 a0 = _mm_loadu_ps(pSrc1);
			__m128 b0 = _mm_loadu_ps(pSrc2);

			__m128 r0 = _mm_sub_ps(a0, b0);
			_mm_store_ps(pDst, r0);
			
			len -= 4;
			pSrc1 += 4;
			pSrc2 += 4;
			pDst += 4;
		}

		for (; len > 0; --len, ++pSrc1, ++pSrc2, ++pDst)
			*pDst = *pSrc1 - *pSrc2;
	}

	template <> void mulC(const float * pSrc, float val, float * pDst, int len)
	{
		__m128 b = _mm_load1_ps(&val);
	
		for (; len >= 8; len-=8, pSrc+=8, pDst+=8)
		{
			__m128 a0 = _mm_loadu_ps(pSrc);
			__m128 a1 = _mm_loadu_ps(pSrc+4);
			
			__m128 r0 = _mm_mul_ps(a0, b);
			__m128 r1 = _mm_mul_ps(a1, b);
			
			_mm_store_ps(pDst, r0);
			_mm_store_ps(pDst+4, r1);
		}
	
		if (len >= 4)
		{
			__m128 a0 = _mm_loadu_ps(pSrc);
			__m128 r0 = _mm_mul_ps(a0, b);
			_mm_store_ps(pDst, r0);
			
			len -= 4;
			pSrc += 4;
			pDst += 4;
		}

		for (; len > 0; --len, ++pSrc, ++pDst)
			*pDst = *pSrc * val;
	}

	template <> void mul(const float * pSrc1, const float * pSrc2, float * pDst, int len)
	{
		for (; len >= 8; len-=8, pSrc1+=8, pSrc2+=8, pDst+=8)
		{
			__m128 a0 = _mm_loadu_ps(pSrc1);
			__m128 a1 = _mm_loadu_ps(pSrc1+4);
			__m128 b0 = _mm_loadu_ps(pSrc2);
			__m128 b1 = _mm_loadu_ps(pSrc2+4);
			
			__m128 r0 = _mm_mul_ps(a0, b0);
			__m128 r1 = _mm_mul_ps(a1, b1);
			
			_mm_store_ps(pDst, r0);
			_mm_store_ps(pDst+4, r1);
		}
	
		if (len >= 4)
		{
			__m128 a0 = _mm_loadu_ps(pSrc1);
			__m128 b0 = _mm_loadu_ps(pSrc2);

			__m128 r0 = _mm_mul_ps(a0, b0);
			_mm_store_ps(pDst, r0);
			
			len -= 4;
			pSrc1 += 4;
			pSrc2 += 4;
			pDst += 4;
		}

		for (; len > 0; --len, ++pSrc1, ++pSrc2, ++pDst)
			*pDst = *pSrc1 * *pSrc2;
	}

	template <> void divC(const float * pSrc, float val, float * pDst, int len)
	{
		__m128 b = _mm_load1_ps(&val);
	
		for (; len >= 8; len-=8, pSrc+=8, pDst+=8)
		{
			__m128 a0 = _mm_loadu_ps(pSrc);
			__m128 a1 = _mm_loadu_ps(pSrc+4);
			
			__m128 r0 = _mm_div_ps(a0, b);
			__m128 r1 = _mm_div_ps(a1, b);
			
			_mm_store_ps(pDst, r0);
			_mm_store_ps(pDst+4, r1);
		}
	
		if (len >= 4)
		{
			__m128 a0 = _mm_loadu_ps(pSrc);
			__m128 r0 = _mm_div_ps(a0, b);
			_mm_store_ps(pDst, r0);
			
			len -= 4;
			pSrc += 4;
			pDst += 4;
		}

		for (; len > 0; --len, ++pSrc, ++pDst)
			*pDst = *pSrc / val;
	}

	template <> void divCRev(const float * pSrc, float val, float * pDst, int len)
	{
		__m128 a = _mm_load1_ps(&val);
	
		for (; len >= 8; len-=8, pSrc+=8, pDst+=8)
		{
			__m128 b0 = _mm_loadu_ps(pSrc);
			__m128 b1 = _mm_loadu_ps(pSrc+4);
			
			__m128 r0 = _mm_div_ps(a, b0);
			__m128 r1 = _mm_div_ps(a, b1);
			
			_mm_store_ps(pDst, r0);
			_mm_store_ps(pDst+4, r1);
		}
	
		if (len >= 4)
		{
			__m128 b0 = _mm_loadu_ps(pSrc);
			__m128 r0 = _mm_div_ps(a, b0);
			_mm_store_ps(pDst, r0);
			
			len -= 4;
			pSrc += 4;
			pDst += 4;
		}

		for (; len > 0; --len, ++pSrc, ++pDst)
			*pDst = val / *pSrc;
	}

	template <> void div(const float * pSrc1, const float * pSrc2, float * pDst, int len)
	{
		for (; len >= 8; len-=8, pSrc1+=8, pSrc2+=8, pDst+=8)
		{
			__m128 a0 = _mm_loadu_ps(pSrc1);
			__m128 a1 = _mm_loadu_ps(pSrc1+4);
			__m128 b0 = _mm_loadu_ps(pSrc2);
			__m128 b1 = _mm_loadu_ps(pSrc2+4);
			
			__m128 r0 = _mm_div_ps(a0, b0);
			__m128 r1 = _mm_div_ps(a1, b1);
			
			_mm_store_ps(pDst, r0);
			_mm_store_ps(pDst+4, r1);
		}
	
		if (len >= 4)
		{
			__m128 a0 = _mm_loadu_ps(pSrc1);
			__m128 b0 = _mm_loadu_ps(pSrc2);

			__m128 r0 = _mm_div_ps(a0, b0);
			_mm_store_ps(pDst, r0);
			
			len -= 4;
			pSrc1 += 4;
			pSrc2 += 4;
			pDst += 4;
		}

		for (; len > 0; --len, ++pSrc1, ++pSrc2, ++pDst)
			*pDst = *pSrc1 / *pSrc2;
	}

	// double

	template <> void addC(const double * pSrc, double val, double * pDst, int len)
	{
		__m128d b = _mm_load1_pd(&val);

		for (; len >= 4; len-=4, pSrc+=4, pDst+=4)
		{
			__m128d a0 = _mm_loadu_pd(pSrc);
			__m128d a1 = _mm_loadu_pd(pSrc+2);

			__m128d r0 = _mm_add_pd(a0, b);
			__m128d r1 = _mm_add_pd(a1, b);

			_mm_store_pd(pDst, r0);
			_mm_store_pd(pDst+2, r1);
 		}
 
		for (; len > 0; --len, ++pSrc, ++pDst)
			*pDst = *pSrc + val;
	}

	template <> void add(const double * pSrc1, const double * pSrc2, double * pDst, int len)
	{
		for (; len >= 4; len-=4, pSrc1+=4, pSrc2+=4, pDst+=4)
		{
			__m128d a0 = _mm_loadu_pd(pSrc1);
			__m128d a1 = _mm_loadu_pd(pSrc1+2);
			__m128d b0 = _mm_loadu_pd(pSrc2);
			__m128d b1 = _mm_loadu_pd(pSrc2+2);

			__m128d r0 = _mm_add_pd(a0, b0);
			__m128d r1 = _mm_add_pd(a1, b1);

			_mm_store_pd(pDst, r0);
			_mm_store_pd(pDst+2, r1);
 		}
 
		for (; len > 0; --len, ++pSrc1, ++pSrc2, ++pDst)
			*pDst = *pSrc1 + *pSrc2;
	}

	template <> void subC(const double * pSrc, double val, double * pDst, int len)
	{
		__m128d b = _mm_load1_pd(&val);

		for (; len >= 4; len-=4, pSrc+=4, pDst+=4)
		{
			__m128d a0 = _mm_loadu_pd(pSrc);
			__m128d a1 = _mm_loadu_pd(pSrc+2);

			__m128d r0 = _mm_sub_pd(a0, b);
			__m128d r1 = _mm_sub_pd(a1, b);

			_mm_store_pd(pDst, r0);
			_mm_store_pd(pDst+2, r1);
 		}
 
		for (; len > 0; --len, ++pSrc, ++pDst)
			*pDst = *pSrc - val;
	}

	template <> void subCRev(const double * pSrc, double val, double * pDst, int len)
	{
		__m128d a = _mm_load1_pd(&val);

		for (; len >= 4; len-=4, pSrc+=4, pDst+=4)
		{
			__m128d b0 = _mm_loadu_pd(pSrc);
			__m128d b1 = _mm_loadu_pd(pSrc+2);

			__m128d r0 = _mm_sub_pd(a, b0);
			__m128d r1 = _mm_sub_pd(a, b1);

			_mm_store_pd(pDst, r0);
			_mm_store_pd(pDst+2, r1);
 		}
 
		for (; len > 0; --len, ++pSrc, ++pDst)
			*pDst = val - *pSrc;
	}

	template <> void sub(const double * pSrc1, const double * pSrc2, double * pDst, int len)
	{
		for (; len >= 4; len-=4, pSrc1+=4, pSrc2+=4, pDst+=4)
		{
			__m128d a0 = _mm_loadu_pd(pSrc1);
			__m128d a1 = _mm_loadu_pd(pSrc1+2);
			__m128d b0 = _mm_loadu_pd(pSrc2);
			__m128d b1 = _mm_loadu_pd(pSrc2+2);

			__m128d r0 = _mm_sub_pd(a0, b0);
			__m128d r1 = _mm_sub_pd(a1, b1);

			_mm_store_pd(pDst, r0);
			_mm_store_pd(pDst+2, r1);
 		}
 
		for (; len > 0; --len, ++pSrc1, ++pSrc2, ++pDst)
			*pDst = *pSrc1 - *pSrc2;
	}

	template <> void mulC(const double * pSrc, double val, double * pDst, int len)
	{
		__m128d b = _mm_load1_pd(&val);

		for (; len >= 4; len-=4, pSrc+=4, pDst+=4)
		{
			__m128d a0 = _mm_loadu_pd(pSrc);
			__m128d a1 = _mm_loadu_pd(pSrc+2);

			__m128d r0 = _mm_mul_pd(a0, b);
			__m128d r1 = _mm_mul_pd(a1, b);

			_mm_store_pd(pDst, r0);
			_mm_store_pd(pDst+2, r1);
 		}
 
		for (; len > 0; --len, ++pSrc, ++pDst)
			*pDst = *pSrc * val;
	}

	template <> void mul(const double * pSrc1, const double * pSrc2, double * pDst, int len)
	{
		for (; len >= 4; len-=4, pSrc1+=4, pSrc2+=4, pDst+=4)
		{
			__m128d a0 = _mm_loadu_pd(pSrc1);
			__m128d a1 = _mm_loadu_pd(pSrc1+2);
			__m128d b0 = _mm_loadu_pd(pSrc2);
			__m128d b1 = _mm_loadu_pd(pSrc2+2);

			__m128d r0 = _mm_mul_pd(a0, b0);
			__m128d r1 = _mm_mul_pd(a1, b1);

			_mm_store_pd(pDst, r0);
			_mm_store_pd(pDst+2, r1);
 		}
 
		for (; len > 0; --len, ++pSrc1, ++pSrc2, ++pDst)
			*pDst = *pSrc1 * *pSrc2;
	}

	template <> void divC(const double * pSrc, double val, double * pDst, int len)
	{
		__m128d b = _mm_load1_pd(&val);

		for (; len >= 4; len-=4, pSrc+=4, pDst+=4)
		{
			__m128d a0 = _mm_loadu_pd(pSrc);
			__m128d a1 = _mm_loadu_pd(pSrc+2);

			__m128d r0 = _mm_div_pd(a0, b);
			__m128d r1 = _mm_div_pd(a1, b);

			_mm_store_pd(pDst, r0);
			_mm_store_pd(pDst+2, r1);
 		}
 
		for (; len > 0; --len, ++pSrc, ++pDst)
			*pDst = *pSrc / val;
	}

	template <> void divCRev(const double * pSrc, double val, double * pDst, int len)
	{
		__m128d a = _mm_load1_pd(&val);

		for (; len >= 4; len-=4, pSrc+=4, pDst+=4)
		{
			__m128d b0 = _mm_loadu_pd(pSrc);
			__m128d b1 = _mm_loadu_pd(pSrc+2);

			__m128d r0 = _mm_div_pd(a, b0);
			__m128d r1 = _mm_div_pd(a, b1);

			_mm_store_pd(pDst, r0);
			_mm_store_pd(pDst+2, r1);
 		}
 
		for (; len > 0; --len, ++pSrc, ++pDst)
			*pDst = val / *pSrc;
	}

	template <> void div(const double * pSrc1, const double * pSrc2, double * pDst, int len)
	{
		for (; len >= 4; len-=4, pSrc1+=4, pSrc2+=4, pDst+=4)
		{
			__m128d a0 = _mm_loadu_pd(pSrc1);
			__m128d a1 = _mm_loadu_pd(pSrc1+2);
			__m128d b0 = _mm_loadu_pd(pSrc2);
			__m128d b1 = _mm_loadu_pd(pSrc2+2);

			__m128d r0 = _mm_div_pd(a0, b0);
			__m128d r1 = _mm_div_pd(a1, b1);

			_mm_store_pd(pDst, r0);
			_mm_store_pd(pDst+2, r1);
 		}
 
		for (; len > 0; --len, ++pSrc1, ++pSrc2, ++pDst)
			*pDst = *pSrc1 / *pSrc2;
	}
}
}

