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

static const unsigned LENGTH = 47;

template<typename T, bool hasDivC = true, bool hasDivCRev = false>
void test_arithm(T value1, T value2)
{
	using std::shared_ptr;
	using std::cerr;
	using std::endl;

	shared_ptr<T> pv1 = shared_ptr<T>(simd::malloc<T>(LENGTH), simd::free<T>);
	shared_ptr<T> pv2 = shared_ptr<T>(simd::malloc<T>(LENGTH), simd::free<T>);
	shared_ptr<T> presult = shared_ptr<T>(simd::malloc<T>(LENGTH+1), simd::free<T>);
	T * v1 = pv1.get();
	T * v2 = pv2.get();
	T * result = presult.get();
	result[LENGTH] = 0x7f;

	bool failed = false;

	simd::set(value1, v1, LENGTH);
	simd::set(value2, v2, LENGTH);

	for (int i=0; i<LENGTH; ++i)
		if (! equal(v1[i], value1) || ! equal(v2[i], value2))
		{
			cerr << "set" << endl;
			failed = true;
		}

	T r = value1 + value2;
	arithm::add(v1, v2, result, LENGTH);
	for (int i=0; i<LENGTH; ++i)
		if (! equal(result[i], r))
		{
			cerr << "add" << endl;
			failed = true;
		}

	r = value1 - value2;
	arithm::sub(v1, v2, result, LENGTH);
	for (int i=0; i<LENGTH; ++i)
		if (! equal(result[i], r))
		{
			cerr << "sub" << endl;
			failed = true;
		}

	r = value1 * value2;
	arithm::mul(v1, v2, result, LENGTH);
	for (int i=0; i<LENGTH; ++i)
		if (! equal(result[i], r))
		{
			cerr << "mul" << endl;
			failed = true;
		}

	r = value1 / value2;
	arithm::div(v1, v2, result, LENGTH);
	for (int i=0; i<LENGTH; ++i)
		if (! equal(result[i], r))
		{
			cerr << "div" << endl;
			failed = true;
		}

#ifndef TEST_FIXED
	r = value1 + value2;
	simd::addC(v1, value2, result, LENGTH);
	for (int i=0; i<LENGTH; ++i)
		if (! equal(result[i], r))
		{
			cerr << "addC" << endl;
			failed = true;
		}

	r = value1 - value2;
	simd::subC(v1, value2, result, LENGTH);
	for (int i=0; i<LENGTH; ++i)
		if (result[i] != r)
		{
			cerr << "subC" << endl;
			failed = true;
		}

	r = value2 - value1;
	simd::subCRev(v1, value2, result, LENGTH);
	for (int i=0; i<LENGTH; ++i)
		if (! equal(result[i], r))
		{
			cerr << "subCRev" << endl;
			failed = true;
		}

	r = value1 * value2;
	simd::mulC(v1, value2, result, LENGTH);
	for (int i=0; i<LENGTH; ++i)
		if (result[i] != r)
		{
			cerr << "mulC" << endl;
			failed = true;
		}

	if (hasDivC)
	{
		r = value1 / value2;
		simd::divC(v1, value2, result, LENGTH);
		for (int i=0; i<LENGTH; ++i)
			if (! equal(result[i], r))
			{
				cerr << "divC" << endl;
				failed = true;
			}
	}

	if (hasDivCRev)
	{
		r = value2 / value1;
		simd::divCRev(v1, value2, result, LENGTH);
		for (int i=0; i<LENGTH; ++i)
			if (! equal(result[i], r))
			{
				cerr << "divCRev" << endl;
				failed = true;
			}
	}
#endif // TEST_FIXED

	if (result[LENGTH] != 0x7f)
	{
		cerr << "length" << endl;
		failed = true;
	}

	if (failed)
		throw __PRETTY_FUNCTION__;
}

template<typename T>
void test_abs()
{
	using std::shared_ptr;
	using std::cerr;
	using std::endl;

	shared_ptr<T> pv = shared_ptr<T>(simd::malloc<T>(LENGTH), simd::free<T>);
	T * v = pv.get();

	bool failed = false;

	T val = 1;
	for (int i=0; i < LENGTH; ++i, val*=-1)
		v[i] = val;

	simd::abs(v, v, LENGTH);

	for (int i=0; i < LENGTH; ++i, val+=1)
		if (v[i] < 0)
		{
			cerr << "abs" << endl;
			failed = true;
		}

	if (failed)
		throw __PRETTY_FUNCTION__;
}

int main()
{
	try
	{
#ifdef TEST_FLOAT

		test_arithm<float, true, true>(2, 1);
#ifdef SIMD_IPP
		test_arithm<double>(1, 2);
#else
		test_arithm<double, true, true>(1, 2);
#endif
		test_abs<float>();
		test_abs<double>();

#else // TEST_FLOAT

		test_arithm<uint8_t>(2, 1);
		test_arithm<int16_t>(2, 1);
		test_arithm<uint16_t>(2, 1);
#ifdef SIMD_IPP
		test_arithm<int32_t, false>(2, 1);
		test_abs<int64_t>();
#else
		test_arithm<int32_t>(2, 1);
#endif
		test_abs<int16_t>();
		test_abs<int32_t>();

#endif // TEST_FLOAT
	}
	catch (const char * msg)
	{
		std::cerr << msg << std::endl;
		return 1;
	}

	return 0;
}

