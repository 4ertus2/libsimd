#include <iostream>
#include <limits>
#include <memory>
#include <cstdint>

#include "simd.h"
#include "compare.h"

template<typename T, typename U>
void test_convert(T valueT, U valueU, unsigned length = 47)
{
	auto pt = std::shared_ptr<T>(simd::malloc<T>(length), simd::free<T>);
	auto pu = std::shared_ptr<U>(simd::malloc<U>(length), simd::free<U>);
	T * t = pt.get();
	U * u = pu.get();

	simd::set(valueT, t, length);

	simd::convert(t, u, length);
	for (int i=0; i<length; ++i)
	{
		if (! equal(u[i], valueU))
			FAIL();
	}
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

void test_float(float value)
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
	catch (const Exception& ex)
	{
		std::cerr << "func: " << ex.func_ << " line: " << ex.line_ << " len: " << ex.length_ << std::endl;
		return 1;
	}

	return 0;
}
