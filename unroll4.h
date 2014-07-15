#ifndef _SIMD_UNROLL4_H_
#define _SIMD_UNROLL4_H_

#include "nosimd.h"

namespace unroll
{
	namespace common
	{
		using nosimd::common::malloc;
		using nosimd::common::free;

		template<typename T> void zero(T * pDst, int len)
		{
			int i = 0;
			int len4 = len & (~0x3);
			for (; i < len4; i+=4)
			{
				pDst[i] = 0;
				pDst[i+1] = 0;
				pDst[i+2] = 0;
				pDst[i+3] = 0;
			}
			for (; i<len; ++i)
				pDst[i] = 0;
		}
	
		template<typename T> void set(T val, T * pDst, int len)
		{
			int i = 0;
			int len4 = len & (~0x3);
			for (; i < len4; i+=4)
			{
				pDst[i] = val;
				pDst[i+1] = val;
				pDst[i+2] = val;
				pDst[i+3] = val;
			}
			for (; i<len; ++i)
				pDst[i] = val;
		}
	
		template<typename T> void copy(const T * pSrc, T * pDst, int len)
		{
			int i = 0;
			int len4 = len & (~0x3);
			for (; i < len4; i+=4)
			{
				pDst[i] = pSrc[i];
				pDst[i+1] = pSrc[i+1];
				pDst[i+2] = pSrc[i+2];
				pDst[i+3] = pSrc[i+3];
			}
			for (; i<len; ++i)
				pDst[i] = pSrc[i];
		}

		using nosimd::common::move;
		using nosimd::common::convert;
	}

	using namespace unroll::common;
	using namespace nosimd::arithmetic;
	using namespace nosimd::power;
	using namespace nosimd::exp_log;
	using namespace nosimd::statistical;
}

#endif

