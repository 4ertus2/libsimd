#pragma once
#include <immintrin.h>

#include "sse.h"

namespace sse
{
namespace internals
{
    template <IntrAvxI::Unary op>
    INLINE void iValDstTail(const __m256i& a, int32_t * pDst, int len)
    {
        __m256i mask = avxTailMask32(len);
        _mm256_maskstore_epi32(pDst, mask, op(a));
    }

    template <IntrAvxI::Unary op>
    INLINE void iPrtDstTail(const int32_t * pSrc, int32_t * pDst, int len)
    {
        __m256i mask = avxTailMask32(len);
        __m256i x = _mm256_maskload_epi32(pSrc, mask);
        _mm256_maskstore_epi32(pDst, mask, op(x));
    }

    template <IntrAvxI::Binary op>
    INLINE void iPrtValDstTail(const int32_t * pSrc, const __m256i& b, int32_t * pDst, int len)
    {
        __m256i mask = avxTailMask32(len);
        __m256i x = _mm256_maskload_epi32(pSrc, mask);
        _mm256_maskstore_epi32(pDst, mask, op(x, b));
    }

    template <IntrAvxI::Binary op>
    INLINE void iPrtPtrDstTail(const int32_t * pSrc1, const int32_t * pSrc2, int32_t * pDst, int len)
    {
        __m256i mask = avxTailMask32(len);
        __m256i x = _mm256_maskload_epi32(pSrc1, mask);
        __m256i y = _mm256_maskload_epi32(pSrc2, mask);
        _mm256_maskstore_epi32(pDst, mask, op(x, y));
    }

    //

    template <IntrAvxI::Unary op>
    INLINE void iValDstTail(const __m256i& a, int64_t * pDst, int len)
    {
        __m256i mask = avxTailMask64(len);
        _mm256_maskstore_epi64((long long int*)pDst, mask, op(a));
    }

    template <IntrAvxI::Unary op>
    INLINE void iPrtDstTail(const int64_t * pSrc, int64_t * pDst, int len)
    {
        __m256i mask = avxTailMask64(len);
        __m256i x = _mm256_maskload_epi64((const long long int*)pSrc, mask);
        _mm256_maskstore_epi64((long long int*)pDst, mask, op(x));
    }

    template <IntrAvxI::Binary op>
    INLINE void iPrtValDstTail(const int64_t * pSrc, const __m256i& b, int64_t * pDst, int len)
    {
        __m256i mask = avxTailMask64(len);
        __m256i x = _mm256_maskload_epi64((const long long int*)pSrc, mask);
        _mm256_maskstore_epi64((long long int*)pDst, mask, op(x, b));
    }

    template <IntrAvxI::Binary op>
    INLINE void iPrtPtrDstTail(const int64_t * pSrc1, const int64_t * pSrc2, int64_t * pDst, int len)
    {
        __m256i mask = avxTailMask64(len);
        __m256i x = _mm256_maskload_epi64((const long long int*)pSrc1, mask);
        __m256i y = _mm256_maskload_epi64((const long long int*)pSrc2, mask);
        _mm256_maskstore_epi64((long long int*)pDst, mask, op(x, y));
    }

    //

    template <typename _T, IntrAvxI::Unary op>
    INLINE void iValDstEnd(__m256i a, _T * pDst, int len)
    {
        int tail = len % avxBlockLen(_T());
        if (tail) {
            pDst += (len-tail);
            iValDstTail<op>(a, pDst, tail);
        }
        _mm256_zeroall();
    }

    template <typename _T, IntrAvxI::Unary op>
    INLINE void iPtrDstEnd(const _T * pSrc, _T * pDst, int len)
    {
        int tail = len % avxBlockLen(_T());
        if (tail) {
            pSrc += (len-tail);
            pDst += (len-tail);
            iPrtDstTail<op>(pSrc, pDst, tail);
        }
        _mm256_zeroall();
    }

    template <typename _T, IntrAvxI::Binary op>
    INLINE void iPtrValDstEnd(const _T * pSrc, const __m256i& b, _T * pDst, int len)
    {
        int tail = len % avxBlockLen(_T());
        if (tail) {
            pSrc += (len-tail);
            pDst += (len-tail);
            iPrtValDstTail<op>(pSrc, b, pDst, tail);
        }
        _mm256_zeroall();
    }

    template <typename _T, IntrAvxI::Binary op>
    INLINE void iPtrPtrDstEnd(const _T * pSrc1, const _T * pSrc2, _T * pDst, int len)
    {
        int tail = len % avxBlockLen(_T());
        if (tail) {
            pSrc1 += (len-tail);
            pSrc2 += (len-tail);
            pDst += (len-tail);
            iPrtPtrDstTail<op>(pSrc1, pSrc2, pDst, tail);
        }
        _mm256_zeroall();
    }

    //

    template <  IntrAvxI::Unary op,
                IntrAvxI::Store store = avx_store_si>
    INLINE void iValDst(__m256i a, __m256i * pDst, int len)
    {
        for (; len >= 4; len-=4, pDst+=4)
        {
            store(pDst, op(a));
            store(pDst+1, op(a));
            store(pDst+2, op(a));
            store(pDst+3, op(a));
        }

        if (len >= 2)
        {
            store(pDst, op(a));
            store(pDst+1, op(a));
            len -= 2; pDst += 2;
        }

        if (len)
            store(pDst, op(a));
    }

    template <  IntrAvxI::Unary op,
                IntrAvxI::Load load = avx_load_si,
                IntrAvxI::Store store = avx_store_si>
    INLINE void iPtrDst(const __m256i * pSrc, __m256i * pDst, int len)
    {
        for (; len >= 4; len-=4, pSrc+=4, pDst+=4)
        {
            store(pDst, op(load(pSrc)));
            store(pDst+1, op(load(pSrc+1)));
            store(pDst+2, op(load(pSrc+2)));
            store(pDst+3, op(load(pSrc+3)));
        }

        if (len >= 2)
        {
            store(pDst, op(load(pSrc)));
            store(pDst+1, op(load(pSrc+1)));
            len -= 2; pSrc += 2; pDst += 2;
        }

        if (len)
            store(pDst, op(load(pSrc)));
    }

    template <  IntrAvxI::Binary op,
                IntrAvxI::Load load = avx_load_si,
                IntrAvxI::Store store = avx_store_si>
    INLINE void iPtrValDst(const __m256i * pSrc, __m256i b, __m256i * pDst, int len)
    {
        for (; len >= 4; len-=4, pSrc+=4, pDst+=4)
        {
            store(pDst, op(load(pSrc), b));
            store(pDst+1, op(load(pSrc+1), b));
            store(pDst+2, op(load(pSrc+2), b));
            store(pDst+3, op(load(pSrc+3), b));
        }

        if (len >= 2)
        {
            store(pDst, op(load(pSrc), b));
            store(pDst+1, op(load(pSrc+1), b));
            len -= 2; pSrc += 2; pDst += 2;
        }

        if (len)
            store(pDst, op(load(pSrc), b));
    }

    template <  IntrAvxI::Binary op,
                IntrAvxI::Load load = avx_load_si,
                IntrAvxI::Store store = avx_store_si>
    INLINE void iPtrPtrDst(const __m256i * pSrc1, const __m256i * pSrc2, __m256i * pDst, int len)
    {
        for (; len >= 4; len-=4, pSrc1+=4, pSrc2+=4, pDst+=4)
        {
            __m256i a0 = load(pSrc1);
            __m256i a1 = load(pSrc1+1);
            __m256i a2 = load(pSrc1+2);
            __m256i a3 = load(pSrc1+3);

            __m256i b0 = load(pSrc2);
            __m256i b1 = load(pSrc2+1);
            __m256i b2 = load(pSrc2+2);
            __m256i b3 = load(pSrc2+3);

            store(pDst, op(a0, b0));
            store(pDst+1, op(a1, b1));
            store(pDst+2, op(a2, b2));
            store(pDst+3, op(a3, b3));
        }

        if (len >= 2)
        {
            store(pDst, op(load(pSrc1), load(pSrc2)));
            store(pDst+1, op(load(pSrc1+1), load(pSrc2+1)));
            len -= 2; pSrc1 += 2; pSrc2 += 2; pDst += 2;
        }

        if (len)
            store(pDst, op(load(pSrc1), load(pSrc2)));
    }

    //

    template <IntrAvxI::Unary op>
    INLINE void valDst(int32_t value, int32_t * pDst, int len)
    {
        __m256i a = _mm256_set1_epi32(value);
        iValDst<op>(a, (__m256i*)pDst, (len>>3));
        iValDstEnd<int32_t, op>(a, pDst, len);
    }

    template <IntrAvxI::Unary op>
    INLINE void ptrDst(const int32_t * pSrc, int32_t * pDst, int len)
    {
        iPtrDst<op>((const __m256i*)pSrc, (__m256i*)pDst, (len>>3));
        iPtrDstEnd<int32_t, op>(pSrc, pDst, len);
    }

    template <IntrAvxI::Binary op>
    INLINE void ptrValDst(const int32_t * pSrc, int32_t val, int32_t * pDst, int len)
    {
        __m256i b = _mm256_set1_epi32(val);
        iPtrValDst<op>((const __m256i*)pSrc, b, (__m256i*)pDst, (len>>3));
        iPtrValDstEnd<int32_t, op>(pSrc, b, pDst, len);
    }

    template <IntrAvxI::Binary op>
    INLINE void ptrPtrDst(const int32_t * pSrc1, const int32_t * pSrc2, int32_t * pDst, int len)
    {
        iPtrPtrDst<op>((const __m256i*)pSrc1, (const __m256i*)pSrc2, (__m256i*)pDst, (len>>3));
        iPtrPtrDstEnd<int32_t, op>(pSrc1, pSrc2, pDst, len);
    }

    //

    template <IntrAvxI::Unary op>
    INLINE void valDst(int64_t value, int64_t * pDst, int len)
    {
        __m256i a = _mm256_set1_epi64x(value);
        iValDst<op>(a, (__m256i*)pDst, (len>>2));
        iValDstEnd<int64_t, op>(a, pDst, len);
    }

    template <IntrAvxI::Unary op>
    INLINE void ptrDst(const int64_t * pSrc, int64_t * pDst, int len)
    {
        iPtrDst<op>((const __m256i*)pSrc, (__m256i*)pDst, (len>>2));
        iPtrDstEnd<int64_t, op>(pSrc, pDst, len);
    }

    template <IntrAvxI::Binary op>
    INLINE void ptrValDst(const int64_t * pSrc, int64_t val, int64_t * pDst, int len)
    {
        __m256i b = _mm256_set1_epi64x(val);
        iPtrValDst<op>((const __m256i*)pSrc, b, (__m256i*)pDst, (len>>2));
        iPtrValDstEnd<int64_t, op>(pSrc, b, pDst, len);
    }

    template <IntrAvxI::Binary op>
    INLINE void ptrPtrDst(const int64_t * pSrc1, const int64_t * pSrc2, int64_t * pDst, int len)
    {
        iPtrPtrDst<op>((const __m256i*)pSrc1, (const __m256i*)pSrc2, (__m256i*)pDst, (len>>2));
        iPtrPtrDstEnd<int64_t, op>(pSrc1, pSrc2, pDst, len);
    }

    //

    template <IntrAvxI::Unary op>
    INLINE void valDst(int16_t value, int16_t * pDst, int len)
    {
        __m256i a = _mm256_set1_epi16(value);
        iValDst<op>(a, (__m256i*)pDst, (len>>4));
        _mm256_zeroall();
    }

    template <IntrAvxI::Unary op>
    INLINE void ptrDst(const int16_t * pSrc, int16_t * pDst, int len)
    {
        iPtrDst<op>((const __m256i*)pSrc, (__m256i*)pDst, (len>>4));
        _mm256_zeroall();
    }

    template <IntrAvxI::Binary op>
    INLINE void ptrValDst(const int16_t * pSrc, int16_t val, int16_t * pDst, int len)
    {
        __m256i b = _mm256_set1_epi16(val);
        iPtrValDst<op>((const __m256i*)pSrc, b, (__m256i*)pDst, (len>>4));
        _mm256_zeroall();
    }

    template <IntrAvxI::Binary op>
    INLINE void ptrPtrDst(const int16_t * pSrc1, const int16_t * pSrc2, int16_t * pDst, int len)
    {
        iPtrPtrDst<op>((const __m256i*)pSrc1, (const __m256i*)pSrc2, (__m256i*)pDst, (len>>4));
        _mm256_zeroall();
    }

    //

    template <IntrAvxI::Unary op>
    INLINE void valDst(int8_t value, int8_t * pDst, int len)
    {
        __m256i a = _mm256_set1_epi8(value);
        iValDst<op>(a, (__m256i*)pDst, (len>>5));
        _mm256_zeroall();
    }

    template <IntrAvxI::Unary op>
    INLINE void ptrDst(const int8_t * pSrc, int8_t * pDst, int len)
    {
        iPtrDst<op>((const __m256i*)pSrc, (__m256i*)pDst, (len>>5));
        _mm256_zeroall();
    }

    template <IntrAvxI::Binary op>
    INLINE void ptrValDst(const int8_t * pSrc, int8_t val, int8_t * pDst, int len)
    {
        __m256i b = _mm256_set1_epi8(val);
        iPtrValDst<op>((const __m256i*)pSrc, b, (__m256i*)pDst, (len>>5));
        _mm256_zeroall();
    }

    template <IntrAvxI::Binary op>
    INLINE void ptrPtrDst(const int8_t * pSrc1, const int8_t * pSrc2, int8_t * pDst, int len)
    {
        iPtrPtrDst<op>((const __m256i*)pSrc1, (const __m256i*)pSrc2, (__m256i*)pDst, (len>>5));
        _mm256_zeroall();
    }
}

namespace common
{
    _SIMD_SSE_SPEC void set(int32_t val, int32_t * pDst, int len)
    {
        internals::valDst<nop>(val, pDst, len);
    }

    _SIMD_SSE_SPEC void set(uint32_t val, uint32_t * pDst, int len)
    {
        set((int32_t)val, (int32_t*)pDst, len);
    }

    _SIMD_SSE_SPEC void set(int64_t val, int64_t * pDst, int len)
    {
        internals::valDst<nop>(val, pDst, len);
    }

    _SIMD_SSE_SPEC void set(uint64_t val, uint64_t * pDst, int len)
    {
        set((int64_t)val, (int64_t*)pDst, len);
    }

    _SIMD_SSE_SPEC void set(int16_t val, int16_t * pDst, int len)
    {
        internals::valDst<nop>(val, pDst, len);
        int tail = len % avxBlockLen(*pDst);
        len -= tail;
        nosimd::common::set(val, pDst+len, tail);
    }

    _SIMD_SSE_SPEC void set(uint16_t val, uint16_t * pDst, int len)
    {
        set((int16_t)val, (int16_t*)pDst, len);
    }

    _SIMD_SSE_SPEC void set(int8_t val, int8_t * pDst, int len)
    {
        internals::valDst<nop>(val, pDst, len);
        int tail = len % avxBlockLen(*pDst);
        len -= tail;
        nosimd::common::set(val, pDst+len, tail);
    }

    _SIMD_SSE_SPEC void set(uint8_t val, uint8_t * pDst, int len)
    {
        set((int8_t)val, (int8_t*)pDst, len);
    }

    //

    _SIMD_SSE_SPEC void copy(const int32_t * pSrc, int32_t * pDst, int len)
    {
        internals::ptrDst<nop>(pSrc, pDst, len);
    }

    _SIMD_SSE_SPEC void copy(const uint32_t * pSrc, uint32_t * pDst, int len)
    {
        copy((const int32_t*)pSrc, (int32_t*)pDst, len);
    }

    _SIMD_SSE_SPEC void copy(const int64_t * pSrc, int64_t * pDst, int len)
    {
        internals::ptrDst<nop>(pSrc, pDst, len);
    }

    _SIMD_SSE_SPEC void copy(const uint64_t * pSrc, uint64_t * pDst, int len)
    {
        copy((const int64_t*)pSrc, (int64_t*)pDst, len);
    }

    _SIMD_SSE_SPEC void copy(const int16_t * pSrc, int16_t * pDst, int len)
    {
        internals::ptrDst<nop>(pSrc, pDst, len);
        int tail = len % avxBlockLen(*pDst);
        len -= tail;
        nosimd::common::copy(pSrc+len, pDst+len, tail);
    }

    _SIMD_SSE_SPEC void copy(const uint16_t * pSrc, uint16_t * pDst, int len)
    {
        copy((const int16_t*)pSrc, (int16_t*)pDst, len);
    }

    _SIMD_SSE_SPEC void copy(const int8_t * pSrc, int8_t * pDst, int len)
    {
        internals::ptrDst<nop>(pSrc, pDst, len);
        int tail = len % avxBlockLen(*pDst);
        len -= tail;
        nosimd::common::copy(pSrc+len, pDst+len, tail);
    }

    _SIMD_SSE_SPEC void copy(const uint8_t * pSrc, uint8_t * pDst, int len)
    {
        copy((const int8_t*)pSrc, (int8_t*)pDst, len);
    }
}

namespace arithmetic
{
    _SIMD_SSE_SPEC void addC(const int32_t * pSrc, int32_t val, int32_t * pDst, int len)
    {
        internals::ptrValDst<_mm256_add_epi32>(pSrc, val, pDst, len);
    }

    _SIMD_SSE_SPEC void subC(const int32_t * pSrc, int32_t val, int32_t * pDst, int len)
    {
        internals::ptrValDst<_mm256_sub_epi32>(pSrc, val, pDst, len);
    }

    _SIMD_SSE_SPEC void mulC(const int32_t * pSrc, int32_t val, int32_t * pDst, int len)
    {
        internals::ptrValDst<_mm256_mullo_epi32>(pSrc, val, pDst, len);
    }

    _SIMD_SSE_SPEC void divC(const int32_t * pSrc, int32_t val, int32_t * pDst, int len)
    {
        nosimd::arithmetic::divC(pSrc, val, pDst, len);
    }


    _SIMD_SSE_SPEC void subCRev(const int32_t * pSrc, int32_t val, int32_t * pDst, int len)
    {
        internals::ptrValDst<IntrAvxI::rev_op<_mm256_sub_epi32>>(pSrc, val, pDst, len);
    }

    _SIMD_SSE_SPEC void divCRev(const int32_t * pSrc, int32_t val, int32_t * pDst, int len)
    {
        nosimd::arithmetic::divCRev(pSrc, val, pDst, len);
    }


    _SIMD_SSE_SPEC void add(const int32_t * pSrc1, const int32_t * pSrc2, int32_t * pDst, int len)
    {
        internals::ptrPtrDst<_mm256_add_epi32>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_SSE_SPEC void sub(const int32_t * pSrc1, const int32_t * pSrc2, int32_t * pDst, int len)
    {
        internals::ptrPtrDst<_mm256_sub_epi32>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_SSE_SPEC void mul(const int32_t * pSrc1, const int32_t * pSrc2, int32_t * pDst, int len)
    {
        internals::ptrPtrDst<_mm256_mullo_epi32>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_SSE_SPEC void div(const int32_t * pSrc1, const int32_t * pSrc2, int32_t * pDst, int len)
    {
        nosimd::arithmetic::div(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_SSE_SPEC void abs(const int32_t * pSrc, int32_t * pDst, int len)
    {
        internals::ptrDst<_mm256_abs_epi32>(pSrc, pDst, len);
    }

    //

    _SIMD_SSE_SPEC void addC(const uint32_t * pSrc, uint32_t val, uint32_t * pDst, int len)
    {
        addC((const int32_t*)pSrc, (int32_t)val, (int32_t*)pDst, len);
    }

    _SIMD_SSE_SPEC void subC(const uint32_t * pSrc, uint32_t val, uint32_t * pDst, int len)
    {
        subC((const int32_t*)pSrc, (int32_t)val, (int32_t*)pDst, len);
    }

    _SIMD_SSE_SPEC void subCRev(const uint32_t * pSrc, uint32_t val, uint32_t * pDst, int len)
    {
        subCRev((const int32_t*)pSrc, (int32_t)val, (int32_t*)pDst, len);
    }

    _SIMD_SSE_SPEC void mulC(const uint32_t * pSrc, uint32_t val, uint32_t * pDst, int len)
    {
#if 1
        mulC((const int32_t*)pSrc, (int32_t)val, (int32_t*)pDst, len);
#else
        nosimd::arithmetic::mulC(pSrc, val, pDst, len);
#endif
    }

    _SIMD_SSE_SPEC void divC(const uint32_t * pSrc, uint32_t val, uint32_t * pDst, int len)
    {
        nosimd::arithmetic::divC(pSrc, val, pDst, len);
    }

    _SIMD_SSE_SPEC void divCRev(const uint32_t * pSrc, uint32_t val, uint32_t * pDst, int len)
    {
        nosimd::arithmetic::divCRev(pSrc, val, pDst, len);
    }


    _SIMD_SSE_SPEC void add(const uint32_t * pSrc1, const uint32_t * pSrc2, uint32_t * pDst, int len)
    {
        add((const int32_t*)pSrc1, (const int32_t*)pSrc2, (int32_t*)pDst, len);
    }

    _SIMD_SSE_SPEC void sub(const uint32_t * pSrc1, const uint32_t * pSrc2, uint32_t * pDst, int len)
    {
        sub((const int32_t*)pSrc1, (const int32_t*)pSrc2, (int32_t*)pDst, len);
    }

    _SIMD_SSE_SPEC void mul(const uint32_t * pSrc1, const uint32_t * pSrc2, uint32_t * pDst, int len)
    {
#if 1
        mul((const int32_t*)pSrc1, (const int32_t*)pSrc2, (int32_t*)pDst, len);
#else
        nosimd::arithmetic::mul(pSrc1, pSrc2, pDst, len);
#endif
    }

    _SIMD_SSE_SPEC void div(const uint32_t * pSrc1, const uint32_t * pSrc2, uint32_t * pDst, int len)
    {
        nosimd::arithmetic::div(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_SSE_SPEC void abs(const uint32_t * pSrc, uint32_t * pDst, int len)
    {
        if (pSrc != pDst)
            copy(pSrc, pDst, len);
    }

    //

    _SIMD_SSE_SPEC void addC(const int64_t * pSrc, int64_t val, int64_t * pDst, int len)
    {
        internals::ptrValDst<_mm256_add_epi64>(pSrc, val, pDst, len);
    }

    _SIMD_SSE_SPEC void subC(const int64_t * pSrc, int64_t val, int64_t * pDst, int len)
    {
        internals::ptrValDst<_mm256_sub_epi64>(pSrc, val, pDst, len);
    }

    _SIMD_SSE_SPEC void subCRev(const int64_t * pSrc, int64_t val, int64_t * pDst, int len)
    {
        internals::ptrValDst<IntrAvxI::rev_op<_mm256_sub_epi64>>(pSrc, val, pDst, len);
    }

    _SIMD_SSE_SPEC void mulC(const int64_t * pSrc, int64_t val, int64_t * pDst, int len)
    {
        nosimd::arithmetic::mulC(pSrc, val, pDst, len);
    }

    _SIMD_SSE_SPEC void divC(const int64_t * pSrc, int64_t val, int64_t * pDst, int len)
    {
        nosimd::arithmetic::divC(pSrc, val, pDst, len);
    }

    _SIMD_SSE_SPEC void divCRev(const int64_t * pSrc, int64_t val, int64_t * pDst, int len)
    {
        nosimd::arithmetic::divCRev(pSrc, val, pDst, len);
    }


    _SIMD_SSE_SPEC void add(const int64_t * pSrc1, const int64_t * pSrc2, int64_t * pDst, int len)
    {
        internals::ptrPtrDst<_mm256_add_epi64>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_SSE_SPEC void sub(const int64_t * pSrc1, const int64_t * pSrc2, int64_t * pDst, int len)
    {
        internals::ptrPtrDst<_mm256_sub_epi64>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_SSE_SPEC void mul(const int64_t * pSrc1, const int64_t * pSrc2, int64_t * pDst, int len)
    {
        nosimd::arithmetic::mul(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_SSE_SPEC void div(const int64_t * pSrc1, const int64_t * pSrc2, int64_t * pDst, int len)
    {
        nosimd::arithmetic::div(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_SSE_SPEC void abs(const int64_t * pSrc, int64_t * pDst, int len)
    {
        nosimd::arithmetic::abs(pSrc, pDst, len);
    }

    //

    _SIMD_SSE_SPEC void addC(const uint64_t * pSrc, uint64_t val, uint64_t * pDst, int len)
    {
        addC((const int64_t*)pSrc, (int64_t)val, (int64_t*)pDst, len);
    }

    _SIMD_SSE_SPEC void subC(const uint64_t * pSrc, uint64_t val, uint64_t * pDst, int len)
    {
        subC((const int64_t*)pSrc, (int64_t)val, (int64_t*)pDst, len);
    }

    _SIMD_SSE_SPEC void subCRev(const uint64_t * pSrc, uint64_t val, uint64_t * pDst, int len)
    {
        subCRev((const int64_t*)pSrc, (int64_t)val, (int64_t*)pDst, len);
    }

    _SIMD_SSE_SPEC void mulC(const uint64_t * pSrc, uint64_t val, uint64_t * pDst, int len)
    {
        nosimd::arithmetic::mulC(pSrc, val, pDst, len);
    }

    _SIMD_SSE_SPEC void divC(const uint64_t * pSrc, uint64_t val, uint64_t * pDst, int len)
    {
        nosimd::arithmetic::divC(pSrc, val, pDst, len);
    }

    _SIMD_SSE_SPEC void divCRev(const uint64_t * pSrc, uint64_t val, uint64_t * pDst, int len)
    {
        nosimd::arithmetic::divCRev(pSrc, val, pDst, len);
    }


    _SIMD_SSE_SPEC void add(const uint64_t * pSrc1, const uint64_t * pSrc2, uint64_t * pDst, int len)
    {
        add((const int64_t*)pSrc1, (const int64_t*)pSrc2, (int64_t*)pDst, len);
    }

    _SIMD_SSE_SPEC void sub(const uint64_t * pSrc1, const uint64_t * pSrc2, uint64_t * pDst, int len)
    {
        sub((const int64_t*)pSrc1, (const int64_t*)pSrc2, (int64_t*)pDst, len);
    }

    _SIMD_SSE_SPEC void mul(const uint64_t * pSrc1, const uint64_t * pSrc2, uint64_t * pDst, int len)
    {
        nosimd::arithmetic::mul(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_SSE_SPEC void div(const uint64_t * pSrc1, const uint64_t * pSrc2, uint64_t * pDst, int len)
    {
        nosimd::arithmetic::div(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_SSE_SPEC void abs(const uint64_t * pSrc, uint64_t * pDst, int len)
    {
        if (pSrc != pDst)
            copy(pSrc, pDst, len);
    }

    //

    _SIMD_SSE_SPEC void addC(const int16_t * pSrc, int16_t val, int16_t * pDst, int len)
    {
        internals::ptrValDst<_mm256_add_epi16>(pSrc, val, pDst, len);
        int tail = len % avxBlockLen(*pDst);
        len -= tail;
        nosimd::arithmetic::addC(pSrc+len, val, pDst+len, tail);
    }

    _SIMD_SSE_SPEC void subC(const int16_t * pSrc, int16_t val, int16_t * pDst, int len)
    {
        internals::ptrValDst<_mm256_sub_epi16>(pSrc, val, pDst, len);
        int tail = len % avxBlockLen(*pDst);
        len -= tail;
        nosimd::arithmetic::subC(pSrc+len, val, pDst+len, tail);
    }

    _SIMD_SSE_SPEC void mulC(const int16_t * pSrc, int16_t val, int16_t * pDst, int len)
    {
        internals::ptrValDst<_mm256_mullo_epi16>(pSrc, val, pDst, len);
        int tail = len % avxBlockLen(*pDst);
        len -= tail;
        nosimd::arithmetic::mulC(pSrc+len, val, pDst+len, tail);
    }

    _SIMD_SSE_SPEC void divC(const int16_t * pSrc, int16_t val, int16_t * pDst, int len)
    {
        nosimd::arithmetic::divC(pSrc, val, pDst, len);
    }


    _SIMD_SSE_SPEC void subCRev(const int16_t * pSrc, int16_t val, int16_t * pDst, int len)
    {
        internals::ptrValDst<IntrAvxI::rev_op<_mm256_sub_epi16>>(pSrc, val, pDst, len);
        int tail = len % avxBlockLen(*pDst);
        len -= tail;
        nosimd::arithmetic::subCRev(pSrc+len, val, pDst+len, tail);
    }

    _SIMD_SSE_SPEC void divCRev(const int16_t * pSrc, int16_t val, int16_t * pDst, int len)
    {
        nosimd::arithmetic::divCRev(pSrc, val, pDst, len);
    }


    _SIMD_SSE_SPEC void add(const int16_t * pSrc1, const int16_t * pSrc2, int16_t * pDst, int len)
    {
        internals::ptrPtrDst<_mm256_add_epi16>(pSrc1, pSrc2, pDst, len);
        int tail = len % avxBlockLen(*pDst);
        len -= tail;
        nosimd::arithmetic::add(pSrc1+len, pSrc2+len, pDst+len, tail);
    }

    _SIMD_SSE_SPEC void sub(const int16_t * pSrc1, const int16_t * pSrc2, int16_t * pDst, int len)
    {
        internals::ptrPtrDst<_mm256_sub_epi16>(pSrc1, pSrc2, pDst, len);
        int tail = len % avxBlockLen(*pDst);
        len -= tail;
        nosimd::arithmetic::sub(pSrc1+len, pSrc2+len, pDst+len, tail);
    }

    _SIMD_SSE_SPEC void mul(const int16_t * pSrc1, const int16_t * pSrc2, int16_t * pDst, int len)
    {
        internals::ptrPtrDst<_mm256_mullo_epi16>(pSrc1, pSrc2, pDst, len);
        int tail = len % avxBlockLen(*pDst);
        len -= tail;
        nosimd::arithmetic::mul(pSrc1+len, pSrc2+len, pDst+len, tail);
    }

    _SIMD_SSE_SPEC void div(const int16_t * pSrc1, const int16_t * pSrc2, int16_t * pDst, int len)
    {
        nosimd::arithmetic::div(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_SSE_SPEC void abs(const int16_t * pSrc, int16_t * pDst, int len)
    {
        internals::ptrDst<_mm256_abs_epi16>(pSrc, pDst, len);
        int tail = len % avxBlockLen(*pDst);
        len -= tail;
        nosimd::arithmetic::abs(pSrc+len, pDst+len, tail);
    }

    //

    _SIMD_SSE_SPEC void addC(const uint16_t * pSrc, uint16_t val, uint16_t * pDst, int len)
    {
        addC((const int16_t*)pSrc, (int16_t)val, (int16_t*)pDst, len);
    }

    _SIMD_SSE_SPEC void subC(const uint16_t * pSrc, uint16_t val, uint16_t * pDst, int len)
    {
        subC((const int16_t*)pSrc, (int16_t)val, (int16_t*)pDst, len);
    }

    _SIMD_SSE_SPEC void subCRev(const uint16_t * pSrc, uint16_t val, uint16_t * pDst, int len)
    {
        subCRev((const int16_t*)pSrc, (int16_t)val, (int16_t*)pDst, len);
    }

    _SIMD_SSE_SPEC void mulC(const uint16_t * pSrc, uint16_t val, uint16_t * pDst, int len)
    {
#if 1
        mulC((const int16_t*)pSrc, (int16_t)val, (int16_t*)pDst, len);
#else
        nosimd::arithmetic::mulC(pSrc, val, pDst, len);
#endif
    }

    _SIMD_SSE_SPEC void divC(const uint16_t * pSrc, uint16_t val, uint16_t * pDst, int len)
    {
        nosimd::arithmetic::divC(pSrc, val, pDst, len);
    }

    _SIMD_SSE_SPEC void divCRev(const uint16_t * pSrc, uint16_t val, uint16_t * pDst, int len)
    {
        nosimd::arithmetic::divCRev(pSrc, val, pDst, len);
    }


    _SIMD_SSE_SPEC void add(const uint16_t * pSrc1, const uint16_t * pSrc2, uint16_t * pDst, int len)
    {
        add((const int16_t*)pSrc1, (const int16_t*)pSrc2, (int16_t*)pDst, len);
    }

    _SIMD_SSE_SPEC void sub(const uint16_t * pSrc1, const uint16_t * pSrc2, uint16_t * pDst, int len)
    {
        sub((const int16_t*)pSrc1, (const int16_t*)pSrc2, (int16_t*)pDst, len);
    }

    _SIMD_SSE_SPEC void mul(const uint16_t * pSrc1, const uint16_t * pSrc2, uint16_t * pDst, int len)
    {
#if 1
        mul((const int16_t*)pSrc1, (const int16_t*)pSrc2, (int16_t*)pDst, len);
#else
        nosimd::arithmetic::mul(pSrc1, pSrc2, pDst, len);
#endif
    }

    _SIMD_SSE_SPEC void div(const uint16_t * pSrc1, const uint16_t * pSrc2, uint16_t * pDst, int len)
    {
        nosimd::arithmetic::div(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_SSE_SPEC void abs(const uint16_t * pSrc, uint16_t * pDst, int len)
    {
        if (pSrc != pDst)
            copy(pSrc, pDst, len);
    }

    //

    _SIMD_SSE_SPEC void addC(const int8_t * pSrc, int8_t val, int8_t * pDst, int len)
    {
        internals::ptrValDst<_mm256_add_epi8>(pSrc, val, pDst, len);
        int tail = len % avxBlockLen(*pDst);
        len -= tail;
        nosimd::arithmetic::addC(pSrc+len, val, pDst+len, tail);
    }

    _SIMD_SSE_SPEC void subC(const int8_t * pSrc, int8_t val, int8_t * pDst, int len)
    {
        internals::ptrValDst<_mm256_sub_epi8>(pSrc, val, pDst, len);
        int tail = len % avxBlockLen(*pDst);
        len -= tail;
        nosimd::arithmetic::subC(pSrc+len, val, pDst+len, tail);
    }

    _SIMD_SSE_SPEC void mulC(const int8_t * pSrc, int8_t val, int8_t * pDst, int len)
    {
        nosimd::arithmetic::mulC(pSrc, val, pDst, len);
    }

    _SIMD_SSE_SPEC void divC(const int8_t * pSrc, int8_t val, int8_t * pDst, int len)
    {
        nosimd::arithmetic::divC(pSrc, val, pDst, len);
    }


    _SIMD_SSE_SPEC void subCRev(const int8_t * pSrc, int8_t val, int8_t * pDst, int len)
    {
        internals::ptrValDst<IntrAvxI::rev_op<_mm256_sub_epi8>>(pSrc, val, pDst, len);
        int tail = len % avxBlockLen(*pDst);
        len -= tail;
        nosimd::arithmetic::subCRev(pSrc+len, val, pDst+len, tail);
    }

    _SIMD_SSE_SPEC void divCRev(const int8_t * pSrc, int8_t val, int8_t * pDst, int len)
    {
        nosimd::arithmetic::divCRev(pSrc, val, pDst, len);
    }


    _SIMD_SSE_SPEC void add(const int8_t * pSrc1, const int8_t * pSrc2, int8_t * pDst, int len)
    {
        internals::ptrPtrDst<_mm256_add_epi8>(pSrc1, pSrc2, pDst, len);
        int tail = len % avxBlockLen(*pDst);
        len -= tail;
        nosimd::arithmetic::add(pSrc1+len, pSrc2+len, pDst+len, tail);
    }

    _SIMD_SSE_SPEC void sub(const int8_t * pSrc1, const int8_t * pSrc2, int8_t * pDst, int len)
    {
        internals::ptrPtrDst<_mm256_sub_epi8>(pSrc1, pSrc2, pDst, len);
        int tail = len % avxBlockLen(*pDst);
        len -= tail;
        nosimd::arithmetic::sub(pSrc1+len, pSrc2+len, pDst+len, tail);
    }

    _SIMD_SSE_SPEC void mul(const int8_t * pSrc1, const int8_t * pSrc2, int8_t * pDst, int len)
    {
        nosimd::arithmetic::mul(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_SSE_SPEC void div(const int8_t * pSrc1, const int8_t * pSrc2, int8_t * pDst, int len)
    {
        nosimd::arithmetic::div(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_SSE_SPEC void abs(const int8_t * pSrc, int8_t * pDst, int len)
    {
        internals::ptrDst<_mm256_abs_epi8>(pSrc, pDst, len);
        int tail = len % avxBlockLen(*pDst);
        len -= tail;
        nosimd::arithmetic::abs(pSrc+len, pDst+len, tail);
    }

    //

    _SIMD_SSE_SPEC void addC(const uint8_t * pSrc, uint8_t val, uint8_t * pDst, int len)
    {
        addC((const int8_t*)pSrc, (int8_t)val, (int8_t*)pDst, len);
    }

    _SIMD_SSE_SPEC void subC(const uint8_t * pSrc, uint8_t val, uint8_t * pDst, int len)
    {
        subC((const int8_t*)pSrc, (int8_t)val, (int8_t*)pDst, len);
    }

    _SIMD_SSE_SPEC void subCRev(const uint8_t * pSrc, uint8_t val, uint8_t * pDst, int len)
    {
        subCRev((const int8_t*)pSrc, (int8_t)val, (int8_t*)pDst, len);
    }

    _SIMD_SSE_SPEC void mulC(const uint8_t * pSrc, uint8_t val, uint8_t * pDst, int len)
    {
        nosimd::arithmetic::mulC(pSrc, val, pDst, len);
    }

    _SIMD_SSE_SPEC void divC(const uint8_t * pSrc, uint8_t val, uint8_t * pDst, int len)
    {
        nosimd::arithmetic::divC(pSrc, val, pDst, len);
    }

    _SIMD_SSE_SPEC void divCRev(const uint8_t * pSrc, uint8_t val, uint8_t * pDst, int len)
    {
        nosimd::arithmetic::divCRev(pSrc, val, pDst, len);
    }


    _SIMD_SSE_SPEC void add(const uint8_t * pSrc1, const uint8_t * pSrc2, uint8_t * pDst, int len)
    {
        add((const int8_t*)pSrc1, (const int8_t*)pSrc2, (int8_t*)pDst, len);
    }

    _SIMD_SSE_SPEC void sub(const uint8_t * pSrc1, const uint8_t * pSrc2, uint8_t * pDst, int len)
    {
        sub((const int8_t*)pSrc1, (const int8_t*)pSrc2, (int8_t*)pDst, len);
    }

    _SIMD_SSE_SPEC void mul(const uint8_t * pSrc1, const uint8_t * pSrc2, uint8_t * pDst, int len)
    {
        nosimd::arithmetic::mul(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_SSE_SPEC void div(const uint8_t * pSrc1, const uint8_t * pSrc2, uint8_t * pDst, int len)
    {
        nosimd::arithmetic::div(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_SSE_SPEC void abs(const uint8_t * pSrc, uint8_t * pDst, int len)
    {
        if (pSrc != pDst)
            copy(pSrc, pDst, len);
    }
}

namespace statistical
{
    _SIMD_SSE_T void min(const _T * pSrc, int len, _T * pMin)
    {
        return nosimd::statistical::min(pSrc, len, pMin);
    }

    _SIMD_SSE_T void max(const _T * pSrc, int len, _T * pMax)
    {
        return nosimd::statistical::max(pSrc, len, pMax);
    }

    _SIMD_SSE_T void minMax(const _T * pSrc, int len, _T * pMin, _T * pMax)
    {
        return nosimd::statistical::minMax(pSrc, len, pMin, pMax);
    }

    _SIMD_SSE_T void sum(const _T * pSrc, int len, _T * pSum)
    {
        return nosimd::statistical::sum(pSrc, len, pSum);
    }

    _SIMD_SSE_T void meanStdDev(const _T * pSrc, int len, _T * pMean, _T * pStdDev)
    {
        return nosimd::statistical::meanStdDev(pSrc, len, pMean, pStdDev);
    }

    _SIMD_SSE_T void dotProd(const _T * pSrc1, const _T * pSrc2, int len, _T * pDp)
    {
        return nosimd::statistical::dotProd(pSrc1, pSrc2, len, pDp);
    }
}
}

