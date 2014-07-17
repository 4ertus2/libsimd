#ifndef _SIMD_SSE_H_
#define _SIMD_SSE_H_

#include "nosimd.h"

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
		using namespace nosimd::common;
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
	
	using namespace sse::common;
	using namespace sse::arithmetic;
}

#endif

