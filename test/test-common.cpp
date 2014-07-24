#include <iostream>
#include <memory>
#include <stdint.h>

#include "simd.h"

static const unsigned LENGTH = 47;

template<typename T>
void test_common()
{
	using std::shared_ptr;
	using std::cerr;
	using std::endl;

	static const T VALUE = 42;

	shared_ptr<T> pa = shared_ptr<T>(simd::malloc<T>(LENGTH+1), simd::free<T>);
	shared_ptr<T> pb = shared_ptr<T>(simd::malloc<T>(LENGTH+1), simd::free<T>);
	T * a = pa.get();
	T * b = pb.get();
	a[LENGTH] = 0x7f;
	b[LENGTH] = 0x7f;

	bool failed = false;

	simd::set(VALUE, a, LENGTH);
	for (unsigned i=0; i<LENGTH; ++i)
		if (a[i] != VALUE)
		{
			cerr << "set" << endl;
			failed = true;
		}

	simd::copy(a, b, LENGTH);
	for (unsigned i=0; i<LENGTH; ++i)
		if (b[i] != VALUE)
		{
			cerr << "copy" << endl;
			failed = true;
		}

	simd::zero(b, LENGTH);
	for (unsigned i=0; i<LENGTH; ++i)
		if (b[i] != 0)
		{
			cerr << "zero" << endl;
			failed = true;
		}

	for (unsigned i=0; i<LENGTH; ++i)
		a[i] = i;
	simd::copy(a, b, LENGTH);
	for (unsigned i=0; i<LENGTH; ++i)
		if (b[i] != i)
		{
			cerr << "copy(2)" << endl;
			failed = true;
		}

	if (a[LENGTH] != 0x7f || b[LENGTH] != 0x7f)
	{
		cerr << "length" << endl;
		failed = true;
	}

	if (failed)
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

