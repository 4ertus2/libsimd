#include <iostream>
#include <memory>
#include <cstdint>

#include "simd.h"
#include "compare.h"

template<typename T>
void test_common(unsigned length, T value = 42)
{
    std::shared_ptr<T> pa = std::shared_ptr<T>(simd::malloc<T>(length+1), simd::free<T>);
    std::shared_ptr<T> pb = std::shared_ptr<T>(simd::malloc<T>(length+1), simd::free<T>);
    T * a = pa.get();
    T * b = pb.get();
    a[length] = 0x7f;
    b[length] = 0x3f;

    simd::set(value, a, length);
    for (unsigned i=0; i<length; ++i)
    {
        if (a[i] != value)
            FAIL();
    }

    simd::copy(a, b, length);
    for (unsigned i=0; i<length; ++i)
    {
        if (b[i] != value)
            FAIL();
    }

    simd::zero(b, length);
    for (unsigned i=0; i<length; ++i)
    {
        if (b[i] != 0)
            FAIL();
    }

    for (unsigned i=0; i<length; ++i)
        a[i] = i;
    simd::copy(a, b, length);
    for (unsigned i=0; i<length; ++i)
    {
        if (b[i] != i)
            FAIL();
    }

    if (a[length] != 0x7f || b[length] != 0x3f)
        FAIL();
}

int main()
{
    try
    {
        for (unsigned len = 0; len < 128; ++len)
        {
#ifndef NO_8_16
            test_common<uint8_t>(len);
            test_common<int8_t>(len);
            test_common<uint16_t>(len);
            test_common<int16_t>(len);
#endif
            test_common<uint32_t>(len);
            test_common<int32_t>(len);
            test_common<uint64_t>(len);
            test_common<int64_t>(len);

            test_common<float>(len);
            test_common<double>(len);
        }
    }
    catch (const Exception& ex)
    {
        std::cerr << "func: " << ex.func_ << " line: " << ex.line_ << " len: " << ex.length_ << std::endl;
        return 1;
    }

    return 0;
}

