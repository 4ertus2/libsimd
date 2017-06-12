#include <iostream>
#include <memory>
#include <cstdint>

#include "simd.h"
#include "compare.h"

template<typename T>
void test_arithm(unsigned length, T value1, T value2)
{
    auto pv1 = std::shared_ptr<T>(simd::malloc<T>(length), simd::free<T>);
    auto pv2 = std::shared_ptr<T>(simd::malloc<T>(length), simd::free<T>);
    auto presult = std::shared_ptr<T>(simd::malloc<T>(length+1), simd::free<T>);
    T * v1 = pv1.get();
    T * v2 = pv2.get();
    T * result = presult.get();
    result[length] = 0x7f;

    simd::set(value1, v1, length);
    simd::set(value2, v2, length);

    for (int i=0; i<length; ++i)
    {
        if (! equal(v1[i], value1) || ! equal(v2[i], value2))
            FAIL();
    }

    T r = value1 + value2;
    arithm::add(v1, v2, result, length);
    for (int i=0; i<length; ++i)
    {
        if (! equal(result[i], r))
            FAIL();
    }

    r = value1 - value2;
    arithm::sub(v1, v2, result, length);
    for (int i=0; i<length; ++i)
    {
        if (! equal(result[i], r))
            FAIL();
    }

    r = value1 * value2;
    arithm::mul(v1, v2, result, length);
    for (int i=0; i<length; ++i)
    {
        if (! equal(result[i], r))
            FAIL();
    }

    r = value1 / value2;
    arithm::div(v1, v2, result, length);
    for (int i=0; i<length; ++i)
    {
        if (! equal(result[i], r))
            FAIL();
    }

#ifndef TEST_FIXED
    r = value1 + value2;
    simd::addC(v1, value2, result, length);
    for (int i=0; i<length; ++i)
    {
        if (! equal(result[i], r))
            FAIL();
    }

    r = value1 - value2;
    simd::subC(v1, value2, result, length);
    for (int i=0; i<length; ++i)
    {
        if (result[i] != r)
            FAIL();
    }

    r = value2 - value1;
    simd::subCRev(v1, value2, result, length);
    for (int i=0; i<length; ++i)
    {
        if (! equal(result[i], r))
            FAIL();
    }

    r = value1 * value2;
    simd::mulC(v1, value2, result, length);
    for (int i=0; i<length; ++i)
    {
        if (result[i] != r)
            FAIL();
    }

    r = value1 / value2;
    simd::divC(v1, value2, result, length);
    for (int i=0; i<length; ++i)
    {
        if (! equal(result[i], r))
            FAIL();
    }

    r = value2 / value1;
    simd::divCRev(v1, value2, result, length);
    for (int i=0; i<length; ++i)
    {
        if (! equal(result[i], r))
            FAIL();
    }
#endif // TEST_FIXED

    if (result[length] != 0x7f)
        FAIL();
}

template<typename T>
void test_abs(unsigned length = 47)
{
    auto pv = std::shared_ptr<T>(simd::malloc<T>(length), simd::free<T>);
    T * v = pv.get();

    T val = 1;
    for (int i=0; i < length; ++i, val*=-1)
        v[i] = val;

    simd::abs(v, v, length);

    for (int i=0; i < length; ++i, val+=1)
    {
        if (v[i] < 0)
            FAIL();
    }
}

int main()
{
    try
    {
        for (unsigned len = 0; len < 128; ++len)
        {
            test_arithm<float>(len, 2, 1);
            test_arithm<double>(len, 1, 2);

            test_abs<float>(len);
            test_abs<double>(len);

#ifndef NO_8_16
            test_arithm<uint8_t>(len, 2, 1);
            test_arithm<int16_t>(len, 2, 1);
            test_arithm<uint16_t>(len, 2, 1);
            test_abs<int16_t>(len);
#endif
            test_arithm<int32_t>(len, 2, 1);
            test_arithm<uint32_t>(len, 2, 1);
            test_abs<int32_t>(len);
            test_abs<uint32_t>(len);

            test_arithm<int64_t>(len, 2, 1);
            test_arithm<uint64_t>(len, 2, 1);
            test_abs<int64_t>(len);
            test_abs<uint64_t>(len);
        }
    }
    catch (const Exception& ex)
    {
        std::cerr << "func: " << ex.func_ << " line: " << ex.line_ << " len: " << ex.length_ << std::endl;
        return 1;
    }

    return 0;
}
