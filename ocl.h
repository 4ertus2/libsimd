#pragma once
#include "nosimd.h"

#ifndef _SIMD_OCL_T
#define _SIMD_OCL_T template<typename _T>
#define _SIMD_OCL_SPEC template <>
#endif

namespace ocl
{
    namespace internals
    {
        int alignedSize(int len);
    }

    namespace common
    {
        _SIMD_OCL_T _T* malloc(int len)
        {
            return new _T[internals::alignedSize(len)];
        }

        _SIMD_OCL_T void free(_T* ptr)
        {
            delete [] ptr;
        }

        using nosimd::common::set;
        using nosimd::common::copy;
        using nosimd::common::zero;
        using nosimd::common::move;
        using nosimd::common::convert;
    }

    namespace arithmetic
    {
        _SIMD_OCL_T void addC(const _T* pSrc, _T val, _T* pDst, int len);
        _SIMD_OCL_T void add(const _T* pSrc1, const _T* pSrc2, _T* pDst, int len);

        _SIMD_OCL_T void subC(const _T* pSrc, _T val, _T* pDst, int len);
        _SIMD_OCL_T void subCRev(const _T* pSrc, _T val, _T* pDst, int len);
        _SIMD_OCL_T void sub(const _T* pSrc1, const _T* pSrc2, _T* pDst, int len);

        _SIMD_OCL_T void mulC(const _T* pSrc, _T val, _T* pDst, int len);
        _SIMD_OCL_T void mul(const _T* pSrc1, const _T* pSrc2, _T* pDst, int len);

        _SIMD_OCL_T void divC(const _T* pSrc, _T val, _T* pDst, int len);
        _SIMD_OCL_T void divCRev(const _T* pSrc, _T val, _T* pDst, int len);
        _SIMD_OCL_T void div(const _T* pSrc1, const _T* pSrc2, _T* pDst, int len);

        _SIMD_OCL_T void abs(const _T* pSrc, _T* pDst, int len);
    }

    using namespace ocl::common;
    using namespace ocl::arithmetic;

    using namespace nosimd::compare;
    using namespace nosimd::power;
    using namespace nosimd::statistical;
    using namespace nosimd::exp_log;
    using namespace nosimd::trigonometric;
}
