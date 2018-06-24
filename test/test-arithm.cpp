#include <iostream>
#include <memory>
#include <cstdint>
#include <limits>

#include "simd.h"
#include "compare.h"

template<typename T>
void test_arithm(unsigned length, T value1, T value2, bool allowTrash = false)
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

    for (unsigned i = 0; i < length; ++i)
    {
        if (! equal(v1[i], value1) || ! equal(v2[i], value2))
            FAIL();
    }

    T r = value1 + value2;
    arithm::add(v1, v2, result, length);
    for (unsigned i = 0; i < length; ++i)
    {
        if (! equal(result[i], r))
            FAIL();
    }

    r = value1 - value2;
    arithm::sub(v1, v2, result, length);
    for (unsigned i = 0; i < length; ++i)
    {
        if (! equal(result[i], r))
            FAIL();
    }

    r = value1 * value2;
    arithm::mul(v1, v2, result, length);
    for (unsigned i = 0; i < length; ++i)
    {
        if (! equal(result[i], r))
            FAIL();
    }

    r = value1 / value2;
    arithm::div(v1, v2, result, length);
    for (unsigned i = 0; i < length; ++i)
    {
        if (! equal(result[i], r))
            FAIL();
    }

#ifndef TEST_FIXED
    r = value1 + value2;
    simd::addC(v1, value2, result, length);
    for (unsigned i = 0; i < length; ++i)
    {
        if (! equal(result[i], r))
            FAIL();
    }

    r = value1 - value2;
    simd::subC(v1, value2, result, length);
    for (unsigned i = 0; i < length; ++i)
    {
        if (result[i] != r)
            FAIL();
    }

    r = value2 - value1;
    simd::subCRev(v1, value2, result, length);
    for (unsigned i = 0; i < length; ++i)
    {
        if (! equal(result[i], r))
            FAIL();
    }

    r = value1 * value2;
    simd::mulC(v1, value2, result, length);
    for (unsigned i = 0; i < length; ++i)
    {
        if (result[i] != r)
            FAIL();
    }

    r = value1 / value2;
    simd::divC(v1, value2, result, length);
    for (unsigned i = 0; i < length; ++i)
    {
        if (! equal(result[i], r))
            FAIL();
    }

    r = value2 / value1;
    simd::divCRev(v1, value2, result, length);
    for (unsigned i = 0; i < length; ++i)
    {
        if (! equal(result[i], r))
            FAIL();
    }
#endif // TEST_FIXED

    if (!allowTrash && result[length] != 0x7f)
        FAIL();
}

template<typename T>
void test_abs(unsigned length = 47)
{
    auto pv = std::shared_ptr<T>(simd::malloc<T>(length), simd::free<T>);
    T * v = pv.get();

    for (int i = 0; i < (int)length; ++i)
        v[i] = -i;

    simd::abs(v, v, length);

    for (int i = 0; i < (int)length; ++i)
    {
        if (v[i] < 0 && v[i] != std::numeric_limits<T>::min())
            FAIL();
    }
}

int main()
{
    try
    {
#ifdef ALLOW_TRASH
        bool allowTrash = true;
#else
        bool allowTrash = false;
#endif
#ifdef MORE_DATA
        unsigned start = 1000;
        unsigned end = 1024 * 1024;
        unsigned inc = 100000;
#else
        unsigned start = 0;
        unsigned end = 128;
        unsigned inc = 1;
#endif
        for (unsigned len = start; len < end; len+=inc)
        {
            test_arithm<float>(len, 2, 1, allowTrash);
            test_abs<float>(len);

            test_arithm<double>(len, 2, 1, allowTrash);
            test_abs<double>(len);
        }

        for (unsigned len = start; len < end; len+=inc)
        {
            test_arithm<int32_t>(len, 2, 1, allowTrash);
            test_arithm<uint32_t>(len, 2, 1, allowTrash);
            test_abs<int32_t>(len);
            test_abs<uint32_t>(len);

            test_arithm<int64_t>(len, 2, 1, allowTrash);
            test_arithm<uint64_t>(len, 2, 1, allowTrash);
            test_abs<int64_t>(len);
            test_abs<uint64_t>(len);
        }

#ifndef NO_8_16
        for (unsigned len = start; len < end; len+=inc)
        {
            test_arithm<uint8_t>(len, 2, 1, allowTrash);
            test_arithm<int16_t>(len, 2, 1, allowTrash);
            test_arithm<uint16_t>(len, 2, 1, allowTrash);
            test_abs<int16_t>(len);
            test_abs<int8_t>(len);
        }
#endif
    }
    catch (const simd::Exception& ex) {
        std::cerr << ex.file() << ":" << ex.line() << " (" << ex.code() << ") " << ex.what() << std::endl;
        return 1;
    }
    catch (const Exception& ex)
    {
        std::cerr << "func: " << ex.func_ << " line: " << ex.line_ << " len: " << ex.length_ << std::endl;
        return 1;
    }

    return 0;
}
