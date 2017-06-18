#include <iostream>
#include <limits>
#include <memory>
#include <cstdint>

#include "simd.h"
#include "compare.h"

template<typename _T, typename _U>
void test_convert(unsigned length, _T value)
{
    auto pt = std::shared_ptr<_T>(simd::malloc<_T>(length), simd::free<_T>);
    auto pu = std::shared_ptr<_U>(simd::malloc<_U>(length), simd::free<_U>);
    _T * t = pt.get();
    _U * u = pu.get();

    simd::set(value, t, length);
    simd::convert(t, u, length);

    for (int i = 0; i < length; ++i)
    {
        if (! equal(u[i], (_U)value))
            FAIL();
    }
}

void test_i8(unsigned len, int value)
{
    test_convert<int8_t, int16_t>(len, value);
    test_convert<int8_t, float>(len, value);
}

void test_u8(unsigned len, int value)
{
    test_convert<uint8_t, float>(len, value);
}

void test_i16(unsigned len, int value)
{
    test_convert<int16_t, int32_t>(len, value);
    test_convert<int32_t, int16_t>(len, value);
    test_convert<int16_t, float>(len, value);
}

void test_u16(unsigned len, int value)
{
    test_convert<uint16_t, float>(len, value);
}

void test_i32(unsigned len, int value)
{
    test_convert<int32_t, float>(len, value);
    test_convert<int32_t, double>(len, value);
    test_convert<float, int32_t>(len, value);
    test_convert<double, int32_t>(len, value);
}

void test_i64(unsigned len, int64_t value)
{
    test_convert<int64_t, double>(len, value);
}

void test_float(unsigned len, float value)
{
    test_convert<float, double>(len, value);
    test_convert<double, float>(len, value);
}

int main()
{
    using std::numeric_limits;

    try
    {
        for (unsigned len = 0; len < 128; ++len)
        {
            test_i8(len, 42);
            test_i8(len, numeric_limits<int8_t>::min());
            test_i8(len, numeric_limits<int8_t>::max());

            test_u8(len, 42);
            test_u8(len, numeric_limits<uint8_t>::min());
            test_u8(len, numeric_limits<uint8_t>::max());

            test_i16(len, 42);
            test_i16(len, numeric_limits<int16_t>::min());
            test_i16(len, numeric_limits<int16_t>::max());

            test_u16(len, 42);
            test_u16(len, numeric_limits<uint16_t>::min());
            test_u16(len, numeric_limits<uint16_t>::max());

            test_i32(len, 42);
            test_i32(len, numeric_limits<int32_t>::min());
            test_i32(len, numeric_limits<int32_t>::max());

            test_i64(len, 42);
            test_i64(len, numeric_limits<int64_t>::min());
            test_i64(len, numeric_limits<int64_t>::max());

            test_float(len, 42);
            test_float(len, numeric_limits<float>::min());
            test_float(len, numeric_limits<float>::max());
            test_float(len, numeric_limits<float>::denorm_min());
            test_float(len, numeric_limits<float>::epsilon());
        }
    }
    catch (const Exception& ex)
    {
        std::cerr << "func: " << ex.func_ << " line: " << ex.line_ << " len: " << ex.length_ << std::endl;
        return 1;
    }

    return 0;
}
