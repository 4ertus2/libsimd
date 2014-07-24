#include <iostream>
#include <limits>
#include <memory>
#include <stdint.h>

#include "simd.h"

static const unsigned LENGTH = 47;

template<typename T, typename U>
void test_convert(T valueT, U valueU)
{
	using std::shared_ptr;

	shared_ptr<T> pt = shared_ptr<T>(simd::malloc<T>(LENGTH), simd::free<T>);
	shared_ptr<U> pu = shared_ptr<U>(simd::malloc<U>(LENGTH), simd::free<U>);
	T * t = pt.get();
	U * u = pu.get();

	simd::set(valueT, t, LENGTH);

	simd::convert(t, u, LENGTH);
	for (int i=0; i<LENGTH; ++i)
		if (u[i] != valueU)
			throw __PRETTY_FUNCTION__;
}

void test_i8(int value)
{
	test_convert<int8_t, int16_t>(value, value);
	test_convert<int8_t, float>(value, value);
}

void test_u8(int value)
{
	test_convert<uint8_t, float>(value, value);
}

void test_i16(int value)
{
	test_convert<int16_t, int32_t>(value, value);
	test_convert<int32_t, int16_t>(value, value);
	test_convert<int16_t, float>(value, value);
}

void test_u16(int value)
{
	test_convert<uint16_t, float>(value, value);
}

void test_i32(int value)
{
	test_convert<int32_t, float>(value, value);
	test_convert<int32_t, double>(value, value);
}

void test_i64(int64_t value)
{
	test_convert<int64_t, double>(value, value);
}

void test_float(int value)
{
	test_convert<float, double>(value, value);
	test_convert<double, float>(value, value);
}

int main()
{
	using std::numeric_limits;

	try
	{
		test_i8(42);
		test_i8(numeric_limits<int8_t>::min());
		test_i8(numeric_limits<int8_t>::max());

		test_u8(42);
		test_u8(numeric_limits<uint8_t>::min());
		test_u8(numeric_limits<uint8_t>::max());

		test_i16(42);
		test_i16(numeric_limits<int16_t>::min());
		test_i16(numeric_limits<int16_t>::max());

		test_u16(42);
		test_u16(numeric_limits<uint16_t>::min());
		test_u16(numeric_limits<uint16_t>::max());

		test_i32(42);
		test_i32(numeric_limits<int32_t>::min());
		test_i32(numeric_limits<int32_t>::max());

		test_i64(42);
		test_i64(numeric_limits<int64_t>::min());
		test_i64(numeric_limits<int64_t>::max());

		test_float(42);
		test_float(numeric_limits<float>::min());
		test_float(numeric_limits<float>::max());
		test_float(numeric_limits<float>::denorm_min());
		test_float(numeric_limits<float>::epsilon());
	}
	catch (const char * msg)
	{
		std::cerr << msg << std::endl;
		return 1;
	}

	return 0;
}

