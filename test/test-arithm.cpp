#include <iostream>
#include <limits>
#include <memory>
#include <stdint.h>

#include "simd.h"

namespace
{
	template <typename T> T epsilon() { return 0; }
}

#if TEST_FIXED
#define TEST_FLOAT
namespace arithm
{
	using namespace ipp::f24;
	using namespace ipp::d53;
}
namespace
{
	#define EPS23 ( 1.19209289e-07f )
	#define EPS52 ( 2.2204460492503131e-016 )

	template <> float epsilon() { return EPS23; }
	template <> double epsilon() { return EPS52; }
}
#else
namespace arithm
{
	using namespace simd;
}
namespace
{
	template <> float epsilon() { return std::numeric_limits<float>::epsilon(); }
	template <> double epsilon() { return std::numeric_limits<double>::epsilon(); }
}
#endif

namespace
{
	template<typename T> bool equal(T x, T y) { return x == y; }

	template<> bool equal(float x, float y)
	{
		float r = x-y;
		if (r < 0)
			r = y-x;
		if (r < epsilon<float>())
			return true;
		return false;
	}

	template<> bool equal(double x, double y)
	{
		double r = x-y;
		if (r < 0)
			r = y-x;
		if (r < epsilon<double>())
			return true;
		return false;
	}
}

static const unsigned LENGTH = 32;

template<typename T, bool hasDivC = true>
int test_arithm(T value1, T value2)
{
	using std::shared_ptr;

	shared_ptr<T> pv1 = shared_ptr<T>(simd::malloc<T>(LENGTH), simd::free<T>);
	shared_ptr<T> pv2 = shared_ptr<T>(simd::malloc<T>(LENGTH), simd::free<T>);
	shared_ptr<T> presult = shared_ptr<T>(simd::malloc<T>(LENGTH), simd::free<T>);
	T * v1 = pv1.get();
	T * v2 = pv2.get();
	T * result = presult.get();

	simd::set(value1, v1, LENGTH);
	simd::set(value2, v2, LENGTH);

	T r = value1 + value2;
	arithm::add(v1, v2, result, LENGTH);
	for (int i=0; i<LENGTH; ++i)
		if (! equal(result[i], r))
			throw __PRETTY_FUNCTION__;

	r = value1 - value2;
	arithm::sub(v1, v2, result, LENGTH);
	for (int i=0; i<LENGTH; ++i)
		if (! equal(result[i], r))
			throw __PRETTY_FUNCTION__;

	r = value1 * value2;
	arithm::mul(v1, v2, result, LENGTH);
	for (int i=0; i<LENGTH; ++i)
		if (! equal(result[i], r))
			throw __PRETTY_FUNCTION__;

	r = value1 / value2;
	arithm::div(v1, v2, result, LENGTH);
	for (int i=0; i<LENGTH; ++i)
		if (! equal(result[i], r))
			throw __PRETTY_FUNCTION__;

#ifndef TEST_FIXED
	r = value1 + value2;
	simd::addC(v1, value2, result, LENGTH);
	for (int i=0; i<LENGTH; ++i)
		if (! equal(result[i], r))
			throw __PRETTY_FUNCTION__;

	r = value1 - value2;
	simd::subC(v1, value2, result, LENGTH);
	for (int i=0; i<LENGTH; ++i)
		if (result[i] != r)
			throw __PRETTY_FUNCTION__;

	r = value2 - value1;
	simd::subCRev(v1, value2, result, LENGTH);
	for (int i=0; i<LENGTH; ++i)
		if (! equal(result[i], r))
			throw __PRETTY_FUNCTION__;

	r = value1 * value2;
	simd::mulC(v1, value2, result, LENGTH);
	for (int i=0; i<LENGTH; ++i)
		if (result[i] != r)
			throw __PRETTY_FUNCTION__;

	if (hasDivC)
	{
		r = value1 / value2;
		simd::divC(v1, value2, result, LENGTH);
		for (int i=0; i<LENGTH; ++i)
			if (! equal(result[i], r))
				throw __PRETTY_FUNCTION__;
	}
#endif // TEST_FIXED

	return 0;
}

int main()
{
	try
	{
#ifdef TEST_FLOAT
		test_arithm<float>(2, 1);
		test_arithm<double>(2, 1);
#else

		test_arithm<uint8_t>(2, 1);
		test_arithm<int16_t>(2, 1);
		test_arithm<uint16_t>(2, 1);
#ifdef SIMD_IPP
		test_arithm<int32_t, false>(2, 1);
#else
		test_arithm<int32_t>(2, 1);
#endif

#endif // TEST_FLOAT
	}
	catch (const char * msg)
	{
		std::cerr << msg << std::endl;
		return 1;
	}

	return 0;
}

