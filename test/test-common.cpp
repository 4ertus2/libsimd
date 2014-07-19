#include <iostream>
#include <memory>
#include <stdint.h>

#include "simd.h"

static const unsigned LENGTH = 39;

template<typename T>
void test_common()
{
	using std::shared_ptr;

	static const T VALUE = 42;

	shared_ptr<T> pa = shared_ptr<T>(simd::malloc<T>(LENGTH), simd::free<T>);
	shared_ptr<T> pb = shared_ptr<T>(simd::malloc<T>(LENGTH), simd::free<T>);
	T * a = pa.get();
	T * b = pb.get();

	simd::set(VALUE, a, LENGTH);
	for (unsigned i=0; i<LENGTH; ++i)
		if (a[i] != VALUE)
			throw __PRETTY_FUNCTION__;

	simd::copy(a, b, LENGTH);
	for (unsigned i=0; i<LENGTH; ++i)
		if (b[i] != VALUE)
			throw __PRETTY_FUNCTION__;

	simd::zero(b, LENGTH);
	for (unsigned i=0; i<LENGTH; ++i)
		if (b[i] != 0)
			throw __PRETTY_FUNCTION__;

	for (unsigned i=0; i<LENGTH; ++i)
		a[i] = i;
	simd::copy(a, b, LENGTH);
	for (unsigned i=0; i<LENGTH; ++i)
		if (b[i] != i)
			throw __PRETTY_FUNCTION__;
}

int main()
{
	try
	{
#ifndef TEST_FLOAT
		test_common<uint8_t>();
		test_common<uint16_t>();
		test_common<uint32_t>();
		test_common<uint64_t>();

		test_common<int8_t>();
		test_common<int16_t>();
		test_common<int32_t>();
		test_common<int64_t>();
#endif
		test_common<float>();
		test_common<double>();
	}
	catch (const char * msg)
	{
		std::cerr << msg << std::endl;
		return 1;
	}

	return 0;
}

