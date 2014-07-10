#ifndef _SIMD_NOSIMD_H_
#define _SIMD_NOSIMD_H_

namespace nosimd
{
	namespace common
	{
		template<typename T> T * malloc(int len) { return new T[len]; }
		template<typename T> void free(T * ptr) { delete ptr; }

		template<typename T> void zero(T * pDst, int len)
		{
			for (int i=0; i<len; ++i)
				pDst[i] = 0;
		}
	
		template<typename T> void set(T val, T * pDst, int len)
		{
			for (int i=0; i<len; ++i)
				pDst[i] = val;
		}
	
		template<typename T> void copy(const T * pSrc, T * pDst, int len)
		{
			for (int i=0; i<len; ++i)
				pDst[i] = pSrc[i];
		}

		template<typename T> void move(const T * pSrc, T * pDst, int len)
		{
			for (int i=0; i < len; ++i)
				pDst[i] = pSrc[i];
		}
	}

	using namespace nosimd::common;
}

#endif

