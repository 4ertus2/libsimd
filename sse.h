#pragma once
#include "nosimd.h"

#ifdef max
#undef max
#endif

#ifdef min
#undef min
#endif

#if __cplusplus >= 201103L
#define INLINE __attribute__((always_inline)) inline
#else
#define INLINE inline
#endif

#ifndef _SIMD_SSE_T
#define _SIMD_SSE_T template<typename _T> inline
#define _SIMD_SSE_TU template<typename _T, typename _U> inline
#define _SIMD_SSE_SPEC template <> INLINE
#endif

namespace sse
{
	namespace common
	{
		_SIMD_SSE_T void set(_T val, _T* pDst, int len);
		_SIMD_SSE_T void copy(const _T* pSrc, _T* pDst, int len);

		template<typename _T> void zero(_T* pDst, int len)
		{
			set<_T>((_T)0, pDst, len);
		}
#if 0
		template<typename _T> _T* malloc(int len)
		{
			return (_T*)_mm_malloc(len, 64);
		}

		template<typename _T> void free(_T* ptr)
		{
			_mm_free(ptr);
		}
#else
		using nosimd::common::malloc;
		using nosimd::common::free;
#endif
		using nosimd::common::move;
		using nosimd::common::convert;
	}

	namespace arithmetic
	{
		_SIMD_SSE_T void addC(const _T* pSrc, _T val, _T* pDst, int len);
		_SIMD_SSE_T void add(const _T* pSrc1, const _T* pSrc2, _T* pDst, int len);

		_SIMD_SSE_T void subC(const _T* pSrc, _T val, _T* pDst, int len);
		_SIMD_SSE_T void subCRev(const _T* pSrc, _T val, _T* pDst, int len);
		_SIMD_SSE_T void sub(const _T* pSrc1, const _T* pSrc2, _T* pDst, int len);

		_SIMD_SSE_T void mulC(const _T* pSrc, _T val, _T* pDst, int len);
		_SIMD_SSE_T void mul(const _T* pSrc1, const _T* pSrc2, _T* pDst, int len);

		_SIMD_SSE_T void divC(const _T* pSrc, _T val, _T* pDst, int len);
		_SIMD_SSE_T void divCRev(const _T* pSrc, _T val, _T* pDst, int len);
		_SIMD_SSE_T void div(const _T* pSrc1, const _T* pSrc2, _T* pDst, int len);

		_SIMD_SSE_T void abs(const _T* pSrc, _T* pDst, int len);
	}

	namespace power
	{
		_SIMD_SSE_T void inv(const _T* pSrc, _T* pDst, int len);
		_SIMD_SSE_T void sqrt(const _T* pSrc, _T* pDst, int len);
		_SIMD_SSE_T void invSqrt(const _T* pSrc, _T* pDst, int len);

		template<typename _T> void sqr(const _T* pSrc, _T* pDst, int len)
		{
			sse::arithmetic::mul<_T>(pSrc, pSrc, pDst, len);
		}
	}

	namespace statistical
	{
		_SIMD_SSE_T void min(const _T* pSrc, int len, _T* pMin);
		_SIMD_SSE_T void max(const _T* pSrc, int len, _T* pMax);

		_SIMD_SSE_T void minMax(const _T* pSrc, int len, _T* pMin, _T* pMax);

		_SIMD_SSE_T void sum(const _T* pSrc, int len, _T* pSum);
		_SIMD_SSE_T void meanStdDev(const _T* pSrc, int len, _T* pMean, _T* pStdDev);

		template<typename _T> void mean(const _T* pSrc, int len, _T* pMean)
		{
			sum(pSrc, len, pMean);
			*pMean /= len;
		}

		template<typename _T> void stdDev(const _T* pSrc, int len, _T* pStdDev)
		{
			_T m;
			meanStdDev(pSrc, len, &m, pStdDev);
		}
#if 0
		_SIMD_SSE_T void normInf(const _T* pSrc, int len, _T* pNorm);
		_SIMD_SSE_T void normL1(const _T* pSrc, int len, _T* pNorm);
		_SIMD_SSE_T void normL2(const _T* pSrc, int len, _T* pNorm);

		_SIMD_SSE_T void normDiffInf(const _T* pSrc1, const _T* pSrc2, int len, _T* pNorm);
		_SIMD_SSE_T void normDiffL1(const _T* pSrc1, const _T* pSrc2, int len, _T* pNorm);
		_SIMD_SSE_T void normDiffL2(const _T* pSrc1, const _T* pSrc2, int len, _T* pNorm);
#endif
		_SIMD_SSE_T void dotProd(const _T* pSrc1, const _T* pSrc2, int len, _T* pDp);
	}

	using namespace sse::common;
	using namespace sse::arithmetic;
	using namespace sse::power;
	using namespace sse::statistical;

	using namespace nosimd::exp_log;
	using namespace nosimd::trigonometric;
}
