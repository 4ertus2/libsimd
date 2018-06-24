#pragma once
#include "nosimd.h"
#include "ocl_kernels.h"

#ifndef _SIMD_OCL_T
#define _SIMD_OCL_T template<typename _T>
#define _SIMD_OCL_SPEC template <> __attribute__((always_inline)) inline
#endif

namespace ocl
{
    namespace internals
    {
        int alignedSize(int len);

        template <typename _KernelT, typename _T>
        void execKernel(const _T * pSrc1, const _T * pSrc2, _T * pDst, int len);

        template <typename _KernelT, typename _T>
        void execKernel(const _T * pSrc, _T val, _T * pDst, int len);
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
        _SIMD_OCL_T void add(const _T* pSrc1, const _T* pSrc2, _T* pDst, int len)
        {
            internals::execKernel<internals::Kernel::Add<_T>>(pSrc1, pSrc2, pDst, len);
        }

        _SIMD_OCL_T void sub(const _T* pSrc1, const _T* pSrc2, _T* pDst, int len)
        {
            internals::execKernel<internals::Kernel::Sub<_T>>(pSrc1, pSrc2, pDst, len);
        }

        _SIMD_OCL_T void mul(const _T* pSrc1, const _T* pSrc2, _T* pDst, int len)
        {
            internals::execKernel<internals::Kernel::Mul<_T>>(pSrc1, pSrc2, pDst, len);
        }

        _SIMD_OCL_T void div(const _T* pSrc1, const _T* pSrc2, _T* pDst, int len)
        {
            internals::execKernel<internals::Kernel::Div<_T>>(pSrc1, pSrc2, pDst, len);
        }

        _SIMD_OCL_T void addC(const _T* pSrc, _T val, _T* pDst, int len)
        {
            internals::execKernel<internals::Kernel::AddC<_T>>(pSrc, val, pDst, len);
        }

        _SIMD_OCL_T void subC(const _T* pSrc, _T val, _T* pDst, int len)
        {
            internals::execKernel<internals::Kernel::SubC<_T>>(pSrc, val, pDst, len);
        }

        _SIMD_OCL_T void subCRev(const _T* pSrc, _T val, _T* pDst, int len)
        {
            internals::execKernel<internals::Kernel::SubCRev<_T>>(pSrc, val, pDst, len);
        }

        _SIMD_OCL_T void mulC(const _T* pSrc, _T val, _T* pDst, int len)
        {
            internals::execKernel<internals::Kernel::MulC<_T>>(pSrc, val, pDst, len);
        }

        _SIMD_OCL_T void divC(const _T* pSrc, _T val, _T* pDst, int len)
        {
            internals::execKernel<internals::Kernel::DivC<_T>>(pSrc, val, pDst, len);
        }

        _SIMD_OCL_T void divCRev(const _T* pSrc, _T val, _T* pDst, int len)
        {
            internals::execKernel<internals::Kernel::DivCRev<_T>>(pSrc, val, pDst, len);
        }

        _SIMD_OCL_T void abs(const _T* pSrc, _T* pDst, int len)
        {
            internals::execKernel<internals::Kernel::Abs<_T>>(pSrc, _T(0), pDst, len);
        }

        _SIMD_OCL_SPEC void abs(const uint8_t * pSrc, uint8_t * pDst, int len)
        {
            if (pSrc != pDst)
                common::copy(pSrc, pDst, len);
        }

        _SIMD_OCL_SPEC void abs(const uint16_t * pSrc, uint16_t * pDst, int len)
        {
            if (pSrc != pDst)
                common::copy(pSrc, pDst, len);
        }

        _SIMD_OCL_SPEC void abs(const uint32_t * pSrc, uint32_t * pDst, int len)
        {
            if (pSrc != pDst)
                common::copy(pSrc, pDst, len);
        }

        _SIMD_OCL_SPEC void abs(const uint64_t * pSrc, uint64_t * pDst, int len)
        {
            if (pSrc != pDst)
                common::copy(pSrc, pDst, len);
        }
    }

    using namespace ocl::common;
    using namespace ocl::arithmetic;

    using namespace nosimd::compare;
    using namespace nosimd::power;
    using namespace nosimd::statistical;
    using namespace nosimd::exp_log;
    using namespace nosimd::trigonometric;
}
