#include <memory>
#include <stdint.h>

#include "simd.h"

static const unsigned LENGTH = 128;

template<typename T>
int test_common()
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
			return 1;

	simd::copy(d, c, LENGTH);
	for (unsigned i=0; i<LENGTH; ++i)
		if (c[i] != VALUE)
			return 1;	

	simd::zero(d, LENGTH);
	for (unsigned i=0; i<LENGTH; ++i)
		if (d[i] != 0)
			return 1;

	return 0;
}

int main()
{
	if (test_common<uint8_t>())
		return 1;
	if (test_common<uint16_t>())
		return 1;
	if (test_common<uint32_t>())
		return 1;
	if (test_common<uint64_t>())
		return 1;

	if (test_common<int8_t>())
		return 1;
	if (test_common<int16_t>())
		return 1;
	if (test_common<int32_t>())
		return 1;
	if (test_common<int64_t>())
		return 1;

	if (test_common<float>())
		return 1;
	if (test_common<double>())
		return 1;

	return 0;
}

