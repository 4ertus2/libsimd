#ifndef _SIMD_SSE_H_
#define _SIMD_SSE_H_

#include "nosimd.h"

#ifdef max
#undef max
#endif

#ifdef min
#undef min
#endif

#ifndef _SIMD_EXT_T
#define _SIMD_EXT_T template<typename _T> extern
#endif

#ifndef _SIMD_EXT_TU
#define _SIMD_EXT_TU template<typename _T, typename _U> extern
#endif

namespace sse
{
	namespace common
	{
		_SIMD_EXT_T void set(_T val, _T* pDst, int len);
		_SIMD_EXT_T void zero(_T* pDst, int len);
		_SIMD_EXT_T void copy(const _T* pSrc, _T* pDst, int len);

		using nosimd::common::malloc;
		using nosimd::common::free;
		using nosimd::common::move;
		using nosimd::common::convert;
	}

	namespace arithmetic
	{
		_SIMD_EXT_T void addC(const _T* pSrc, _T val, _T* pDst, int len);
		_SIMD_EXT_T void add(const _T* pSrc1, const _T* pSrc2, _T* pDst, int len);

		_SIMD_EXT_T void subC(const _T* pSrc, _T val, _T* pDst, int len);
		_SIMD_EXT_T void subCRev(const _T* pSrc, _T val, _T* pDst, int len);
		_SIMD_EXT_T void sub(const _T* pSrc1, const _T* pSrc2, _T* pDst, int len);

		_SIMD_EXT_T void mulC(const _T* pSrc, _T val, _T* pDst, int len);
		_SIMD_EXT_T void mul(const _T* pSrc1, const _T* pSrc2, _T* pDst, int len);

		_SIMD_EXT_T void divC(const _T* pSrc, _T val, _T* pDst, int len);
		_SIMD_EXT_T void divCRev(const _T* pSrc, _T val, _T* pDst, int len);
		_SIMD_EXT_T void div(const _T* pSrc1, const _T* pSrc2, _T* pDst, int len);

		_SIMD_EXT_T void abs(const _T* pSrc, _T* pDst, int len);
	}

	namespace power
	{
		_SIMD_EXT_T void inv(const _T* pSrc, _T* pDst, int len);
		_SIMD_EXT_T void sqr(const _T* pSrc, _T* pDst, int len) { sse::arithmetic::mul<_T>(pSrc, pSrc, pDst, len); }
		_SIMD_EXT_T void sqrt(const _T* pSrc, _T* pDst, int len);
		_SIMD_EXT_T void invSqrt(const _T* pSrc, _T* pDst, int len);
	}

	namespace statistical
	{
		_SIMD_EXT_T void min(const _T* pSrc, int len, _T* pMin);
		_SIMD_EXT_T void max(const _T* pSrc, int len, _T* pMax);
		_SIMD_EXT_T void sum(const _T* pSrc, int len, _T* pSum);
	}

	using namespace sse::common;
	using namespace sse::arithmetic;
	using namespace sse::power;
	using namespace sse::statistical;
}

#endif

