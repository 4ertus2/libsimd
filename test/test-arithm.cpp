#include <iostream>
#include <memory>
#include <stdint.h>

#include "simd.h"

static const unsigned LENGTH = 32;

template<typename T, bool hasDivC>
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
	simd::add(v1, v2, result, LENGTH);
	for (int i=0; i<LENGTH; ++i)
		if (result[i] != r)
			throw __PRETTY_FUNCTION__;
			
	r = value1 - value2;
	simd::sub(v1, v2, result, LENGTH);
	for (int i=0; i<LENGTH; ++i)
		if (result[i] != r)
			throw __PRETTY_FUNCTION__;

	r = value1 * value2;
	simd::mul(v1, v2, result, LENGTH);
	for (int i=0; i<LENGTH; ++i)
		if (result[i] != r)
			throw __PRETTY_FUNCTION__;

	r = value1 / value2;
	simd::div(v1, v2, result, LENGTH);
	for (int i=0; i<LENGTH; ++i)
		if (result[i] != r)
			throw __PRETTY_FUNCTION__;

	// xC

	r = value1 + value2;
	simd::addC(v1, value2, result, LENGTH);
	for (int i=0; i<LENGTH; ++i)
		if (result[i] != r)
			throw __PRETTY_FUNCTION__;
			
	r = value1 - value2;
	simd::subC(v1, value2, result, LENGTH);
	for (int i=0; i<LENGTH; ++i)
		if (result[i] != r)
			throw __PRETTY_FUNCTION__;

	r = value2 - value1;
	simd::subCRev(v1, value2, result, LENGTH);
	for (int i=0; i<LENGTH; ++i)
		if (result[i] != r)
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
			if (result[i] != r)
				throw __PRETTY_FUNCTION__;
	}

	return 0;
}

int main()
{
	try
	{
		test_arithm<uint8_t, true>(2, 1);
		test_arithm<int16_t, true>(2, 1);
		test_arithm<uint16_t, true>(2, 1);
		test_arithm<int32_t, false>(2, 1);
	}
	catch (const char * msg)
	{
		std::cerr << msg << std::endl;
		return 1;
	}

	return 0;
}

