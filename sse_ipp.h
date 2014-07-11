#ifndef _SIMD_SSE_IPP_H_
#define _SIMD_SSE_IPP_H_

#ifdef max
#undef max
#endif

#ifdef min
#undef min
#endif

#define _EXT_T_ template<typename _T> extern
#define _EXT_T_U_ template<typename _T, typename _U> extern

namespace ipp
{
	namespace common
	{
		void ipp_free(void* ptr);
		template<typename T> void free(T * ptr) { ipp_free((void*)ptr); }
	
		_EXT_T_ _T* malloc(int len);

		_EXT_T_ void zero(_T* pDst, int len);
		_EXT_T_ void set(_T val, _T* pDst, int len);
		_EXT_T_ void copy(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void move(const _T* pSrc, _T* pDst, int len);

		_EXT_T_U_ void convert(const _T* pSrc, _U* pDst, int len);
	}

	namespace logical_n_shift
	{
		// TODO
	}

	namespace arithmetic
	{
		_EXT_T_ void addC(const _T* pSrc, _T val, _T* pDst, int len);
		_EXT_T_ void add(const _T* pSrc1, const _T* pSrc2, _T* pDst, int len);

		_EXT_T_ void subC(const _T* pSrc, _T val, _T* pDst, int len);
		_EXT_T_ void subCRev(const _T* pSrc, _T val, _T* pDst, int len);
		_EXT_T_ void sub(const _T* pSrc1, const _T* pSrc2, _T* pDst, int len);

		_EXT_T_ void mulC(const _T* pSrc, _T val, _T* pDst, int len);
		_EXT_T_ void mul(const _T* pSrc1, const _T* pSrc2, _T* pDst, int len);

		_EXT_T_ void divC(const _T* pSrc, _T val, _T* pDst, int len);
		_EXT_T_ void divCRev(const _T* pSrc, _T val, _T* pDst, int len);
		_EXT_T_ void div(const _T* pSrc1, const _T* pSrc2, _T* pDst, int len);

		_EXT_T_ void abs(const _T* pSrc, _T* pDst, int len);
	}

	namespace power
	{
		_EXT_T_ void sqr(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void sqrt(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void cubrt(const _T* pSrc, _T* pDst, int len);
	}

	namespace exp_log
	{
		_EXT_T_ void exp(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void ln(const _T* pSrc, _T* pDst, int len);
	}

	namespace threshold
	{
		_EXT_T_ void threshold_LT(const _T* pSrc, _T* pDst, int len, _T level);
		_EXT_T_ void threshold_GT(const _T* pSrc, _T* pDst, int len, _T level);
		_EXT_T_ void threshold_LTAbs(const _T* pSrc, _T* pDst, int len, _T level);
		_EXT_T_ void threshold_GTAbs(const _T* pSrc, _T* pDst, int len, _T level);
	}

	namespace statistical
	{
		_EXT_T_ void sum(const _T* pSrc, int len, _T* pSum);
		_EXT_T_ void mean(const _T* pSrc, int len, _T* pMean);

		_EXT_T_ void max(const _T* pSrc, int len, _T* pMax);
		_EXT_T_ void maxIndx(const _T* pSrc, int len, _T* pMax, int* pIndx);
		_EXT_T_ void maxAbs(const _T* pSrc, int len, _T* pMaxAbs);
		_EXT_T_ void maxAbsIndx(const _T* pSrc, int len, _T* pMaxAbs, int* pIndx);

		_EXT_T_ void min(const _T* pSrc, int len, _T* pMin);
		_EXT_T_ void minIndx(const _T* pSrc, int len, _T* pMin, int* pIndx);
		_EXT_T_ void minAbs(const _T* pSrc, int len, _T* pMinAbs);
		_EXT_T_ void minAbsIndx(const _T* pSrc, int len, _T* pMinAbs, int* pIndx);

		_EXT_T_ void minMax(const _T* pSrc, int len, _T* pMin, _T* pMax);
		_EXT_T_ void minMaxIndx(const _T* pSrc, int len,
			_T* pMin, int* pMinIndx, _T* pMax, int* pMaxIndx);
	}

	namespace rounding
	{
		_EXT_T_ void floor(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void ceil(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void trunc(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void round(const _T* pSrc, _T* pDst, int len);
	}

	using namespace ipp::common;
	using namespace ipp::threshold;
	using namespace ipp::statistical;
	using namespace ipp::rounding;

	using namespace ipp::arithmetic;
	using namespace ipp::power;
	using namespace ipp::exp_log;
}

//
// Fixed-Accuracy Arithmetic Functions
//

// ============================================================================
// float -> A24, double -> A53
namespace ipp_f24_d53
{
	namespace arithmetic
	{
		_EXT_T_ void add(const _T* pSrc1, const _T* pSrc2, _T* pDst, int len);
		_EXT_T_ void sub(const _T* pSrc1, const _T* pSrc2, _T* pDst, int len);
		_EXT_T_ void mul(const _T* pSrc1, const _T* pSrc2, _T* pDst, int len);
		_EXT_T_ void div(const _T* pSrc1, const _T* pSrc2, _T* pDst, int len);
		_EXT_T_ void abs(const _T* pSrc, _T* pDst, int len);
	}

	namespace power
	{
		_EXT_T_ void inv(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void sqr(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void sqrt(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void cbrt(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void invSqrt(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void invCbrt(const _T* pSrc, _T* pDst, int len);

		_EXT_T_ void powx(const _T* pSrc1, const _T ConstValue, _T* pDst, int len);
		_EXT_T_ void pow2o3(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void pow3o2(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void pow(const _T* pSrc1, const _T* pSrc2, _T* pDst, int len);
		_EXT_T_ void hypot(const _T* pSrc1, const _T* pSrc2, _T* pDst, int len); // sqtr(a^2+b^2)
	}

	namespace exp_log
	{
		_EXT_T_ void exp(const _T* pSrc, _T* pDst, int len);
		//_EXT_T_ void expm1(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void ln(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void log10(const _T* pSrc, _T* pDst, int len);
		//_EXT_T_ void log1p(const _T* pSrc, _T* pDst, int len);
	}

	namespace trigonometric
	{
		_EXT_T_ void cos(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void sin(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void sinCos(const _T* pSrc, _T* pDst1, _T* pDst2, int len);
		_EXT_T_ void tan(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void acos(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void asin(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void atan(const _T* pSrc, _T* pDst, int len);
		//_EXT_T_ void atan2(const _T* pSrc1, const _T* pSrc2, _T* pDst, int len);
	}

	namespace hyperbolic
	{
		// TODO
	}

	namespace special
	{
		_EXT_T_ void erf(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void erfc(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void cdfNorm(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void erfInv(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void erfcInv(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void cdfNormInv(const _T* pSrc, _T* pDst, int len);
	}

	using namespace ipp::common;
	using namespace ipp::threshold;
	using namespace ipp::statistical;
	using namespace ipp::rounding;

	using namespace ipp_f24_d53::arithmetic;
	using namespace ipp_f24_d53::power;
	using namespace ipp_f24_d53::exp_log;
	using namespace ipp_f24_d53::trigonometric;
	using namespace ipp_f24_d53::hyperbolic;
	using namespace ipp_f24_d53::special;
}

// ============================================================================
// float -> A21, double -> A50
namespace ipp_f21_d50
{
	namespace arithmetic
	{
		_EXT_T_ void add(const _T* pSrc1, const _T* pSrc2, _T* pDst, int len);
		_EXT_T_ void sub(const _T* pSrc1, const _T* pSrc2, _T* pDst, int len);
		_EXT_T_ void mul(const _T* pSrc1, const _T* pSrc2, _T* pDst, int len);
		_EXT_T_ void div(const _T* pSrc1, const _T* pSrc2, _T* pDst, int len);
		_EXT_T_ void abs(const _T* pSrc, _T* pDst, int len);
	}

	namespace power
	{
		_EXT_T_ void inv(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void sqr(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void sqrt(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void cbrt(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void invSqrt(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void invCbrt(const _T* pSrc, _T* pDst, int len);

		_EXT_T_ void powx(const _T* pSrc1, const _T ConstValue, _T* pDst, int len);
		_EXT_T_ void pow2o3(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void pow3o2(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void pow(const _T* pSrc1, const _T* pSrc2, _T* pDst, int len);
		_EXT_T_ void hypot(const _T* pSrc1, const _T* pSrc2, _T* pDst, int len); // sqtr(a^2+b^2)
	}

	namespace exp_log
	{
		_EXT_T_ void exp(const _T* pSrc, _T* pDst, int len);
		//_EXT_T_ void expm1(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void ln(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void log10(const _T* pSrc, _T* pDst, int len);
		//_EXT_T_ void log1p(const _T* pSrc, _T* pDst, int len);
	}

	namespace trigonometric
	{
		_EXT_T_ void cos(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void sin(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void sinCos(const _T* pSrc, _T* pDst1, _T* pDst2, int len);
		_EXT_T_ void tan(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void acos(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void asin(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void atan(const _T* pSrc, _T* pDst, int len);
		//_EXT_T_ void atan2(const _T* pSrc1, const _T* pSrc2, _T* pDst, int len);
	}

	namespace hyperbolic
	{
		// TODO
	}

	namespace special
	{
		_EXT_T_ void erf(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void erfc(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void cdfNorm(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void erfInv(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void erfcInv(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void cdfNormInv(const _T* pSrc, _T* pDst, int len);
	}

	using namespace ipp::common;
	using namespace ipp::threshold;
	using namespace ipp::statistical;
	using namespace ipp::rounding;

	using namespace ipp_f21_d50::arithmetic;
	using namespace ipp_f21_d50::power;
	using namespace ipp_f21_d50::exp_log;
	using namespace ipp_f21_d50::trigonometric;
	using namespace ipp_f21_d50::hyperbolic;
	using namespace ipp_f21_d50::special;
}

// ============================================================================
// float -> A11, double -> A26
namespace ipp_f11_d26
{
	namespace arithmetic
	{
		_EXT_T_ void div(const _T* pSrc1, const _T* pSrc2, _T* pDst, int len);
		_EXT_T_ void hypot(const _T* pSrc1, const _T* pSrc2, _T* pDst, int len); // sqtr(a^2+b^2)
		_EXT_T_ void inv(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void sqrt(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void cbrt(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void invSqrt(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void invCbrt(const _T* pSrc, _T* pDst, int len);

		_EXT_T_ void powx(const _T* pSrc1, const _T ConstValue, _T* pDst, int len);
		_EXT_T_ void pow2o3(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void pow3o2(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void pow(const _T* pSrc1, const _T* pSrc2, _T* pDst, int len);

		_EXT_T_ void exp(const _T* pSrc, _T* pDst, int len);
		//_EXT_T_ void expm1(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void ln(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void log10(const _T* pSrc, _T* pDst, int len);
		//_EXT_T_ void log1p(const _T* pSrc, _T* pDst, int len);
	}

	namespace trigonometric
	{
		_EXT_T_ void cos(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void sin(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void sinCos(const _T* pSrc, _T* pDst1, _T* pDst2, int len);
		_EXT_T_ void tan(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void acos(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void asin(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void atan(const _T* pSrc, _T* pDst, int len);
		//_EXT_T_ void atan2(const _T* pSrc1, const _T* pSrc2, _T* pDst, int len);
	}

	namespace hyperbolic
	{
		// TODO
	}

	namespace special
	{
		_EXT_T_ void erf(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void erfc(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void cdfNorm(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void erfInv(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void erfcInv(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void cdfNormInv(const _T* pSrc, _T* pDst, int len);
	}

	using namespace ipp::common;
	using namespace ipp::threshold;
	using namespace ipp::statistical;
	using namespace ipp::rounding;

	using namespace ipp_f11_d26::arithmetic;
	using namespace ipp_f11_d26::trigonometric;
	using namespace ipp_f11_d26::hyperbolic;
	using namespace ipp_f11_d26::special;
}

#endif

