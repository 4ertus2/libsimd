#include <iostream>
#include <memory>
#include <stdint.h>

#include "simd.h"

static const unsigned LENGTH = 32;

template<typename T>
void test_common()
{
	using std::shared_ptr;

	static const T VALUE = 42;

	shared_ptr<T> dt1 = shared_ptr<T>(simd::malloc<T>(LENGTH), simd::free<T>);
	shared_ptr<T> dt2 = shared_ptr<T>(simd::malloc<T>(LENGTH), simd::free<T>);
	T * d = dt1.get();
	T * c = dt2.get();

	simd::set(VALUE, d, LENGTH);
	for (unsigned i=0; i<LENGTH; ++i)
		if (d[i] != VALUE)
			throw __PRETTY_FUNCTION__;

	simd::copy(d, c, LENGTH);
	for (unsigned i=0; i<LENGTH; ++i)
		if (c[i] != VALUE)
			throw __PRETTY_FUNCTION__;

	simd::zero(d, LENGTH);
	for (unsigned i=0; i<LENGTH; ++i)
		if (d[i] != 0)
			throw __PRETTY_FUNCTION__;
}

int main()
{
	try
	{
		test_common<uint8_t>();
		test_common<uint16_t>();
		test_common<uint32_t>();
		test_common<uint64_t>();

		test_common<int8_t>();
		test_common<int16_t>();
		test_common<int32_t>();
		test_common<int64_t>();

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

