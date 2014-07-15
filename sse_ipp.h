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
		_EXT_T_ void div(const _T* pSrc1, const _T* pSrc2, _T* pDst, int len);

		_EXT_T_ void abs(const _T* pSrc, _T* pDst, int len);
		
		namespace f21
		{
			void div(const float * pSrc1, const float * pSrc2, float * pDst, int len);
		}
		namespace f24
		{
			void add(const float * pSrc1, const float * pSrc2, float * pDst, int len);
			void sub(const float * pSrc1, const float * pSrc2, float * pDst, int len);
			void mul(const float * pSrc1, const float * pSrc2, float * pDst, int len);
			void div(const float * pSrc1, const float * pSrc2, float * pDst, int len);
			void abs(const float * pSrc, float * pDst, int len);
		}
		namespace d50
		{
			void div(const double * pSrc1, const double * pSrc2, double * pDst, int len);
		}
		namespace d53
		{
			void add(const double * pSrc1, const double * pSrc2, double * pDst, int len);
			void sub(const double * pSrc1, const double * pSrc2, double * pDst, int len);
			void mul(const double * pSrc1, const double * pSrc2, double * pDst, int len);
			void div(const double * pSrc1, const double * pSrc2, double * pDst, int len);
			void abs(const double * pSrc, double * pDst, int len);
		}
	}

	namespace power
	{
		_EXT_T_ void inv(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void sqr(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void sqrt(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void powx(const _T* pSrc, const _T constValue, _T* pDst, int len);
		_EXT_T_ void pow(const _T* pSrc1, const _T* pSrc2, _T* pDst, int len);
#if 0
		_EXT_T_ void cbrt(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void pow2o3(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void pow3o2(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void hypot(const _T* pSrc1, const _T* pSrc2, _T* pDst, int len); // sqtr(a^2+b^2)
#endif
		
		namespace f21
		{
			void inv(const float * pSrc, float * pDst, int len);
			void sqrt(const float * pSrc, float * pDst, int len);
		}
		namespace f24
		{
			void inv(const float * pSrc, float * pDst, int len);
			void sqr(const float * pSrc, float * pDst, int len);
			void sqrt(const float * pSrc, float * pDst, int len);
		}
		namespace d50
		{
			void inv(const double * pSrc, double * pDst, int len);
			void sqrt(const double * pSrc, double * pDst, int len);
		}
		namespace d53
		{
			void inv(const double * pSrc, double * pDst, int len);
			void sqr(const double * pSrc, double * pDst, int len);
			void sqrt(const double * pSrc, double * pDst, int len);
		}
	}

	namespace exp_log
	{
		_EXT_T_ void exp(const _T* pSrc, _T* pDst, int len);
		_EXT_T_ void ln(const _T* pSrc, _T* pDst, int len);

		namespace f21
		{
			void exp(const float * pSrc, float * pDst, int len);
			void ln(const float * pSrc, float * pDst, int len);
		}
		namespace f24
		{
			void exp(const float * pSrc, float * pDst, int len);
			void ln(const float * pSrc, float * pDst, int len);
		}
		namespace d50
		{
			void exp(const double * pSrc, double * pDst, int len);
			void ln(const double * pSrc, double * pDst, int len);
		}
		namespace d53
		{
			void exp(const double * pSrc, double * pDst, int len);
			void ln(const double * pSrc, double * pDst, int len);
		}
	}

	namespace statistical
	{
		_EXT_T_ void max(const _T* pSrc, int len, _T* pMax);
		_EXT_T_ void maxIndx(const _T* pSrc, int len, _T* pMax, int* pIndx);

		_EXT_T_ void min(const _T* pSrc, int len, _T* pMin);
		_EXT_T_ void minIndx(const _T* pSrc, int len, _T* pMin, int* pIndx);

		_EXT_T_ void minMax(const _T* pSrc, int len, _T* pMin, _T* pMax);
		_EXT_T_ void minMaxIndx(const _T* pSrc, int len, _T* pMin, int* pMinIndx, _T* pMax, int* pMaxIndx);
		
		_EXT_T_U_ void sum(const _T* pSrc, int len, _U* pSum);
		_EXT_T_ void mean(const _T* pSrc, int len, _T* pMean);
		_EXT_T_ void stdDev(const _T* pSrc, int len, _T* pStdDev);
		_EXT_T_ void meanStdDev(const _T* pSrc, int len, _T* pMean, _T* pStdDev);

		_EXT_T_U_ void normInf(const _T* pSrc, int len, _U* pNorm);
		_EXT_T_U_ void normL1(const _T* pSrc, int len, _U* pNorm);
		_EXT_T_U_ void normL2(const _T* pSrc, int len, _U* pNorm);
		
		_EXT_T_U_ void dotProd(const _T* pSrc1, const _T* pSrc2, int len, _U* pDp);
	}

	namespace trigonometric
	{
		namespace f21
		{
			void cos(const float * pSrc, float * pDst, int len);
			void sin(const float * pSrc, float * pDst, int len);
			void tan(const float * pSrc, float * pDst, int len);
			void acos(const float * pSrc, float * pDst, int len);
			void asin(const float * pSrc, float * pDst, int len);
			void atan(const float * pSrc, float * pDst, int len);
		}
		namespace f24
		{
			void cos(const float * pSrc, float * pDst, int len);
			void sin(const float * pSrc, float * pDst, int len);
			void tan(const float * pSrc, float * pDst, int len);
			void acos(const float * pSrc, float * pDst, int len);
			void asin(const float * pSrc, float * pDst, int len);
			void atan(const float * pSrc, float * pDst, int len);
		}
		namespace d50
		{
			void cos(const double * pSrc, double * pDst, int len);
			void sin(const double * pSrc, double * pDst, int len);;
			void tan(const double * pSrc, double * pDst, int len);
			void acos(const double * pSrc, double * pDst, int len);
			void asin(const double * pSrc, double * pDst, int len);
			void atan(const double * pSrc, double * pDst, int len);
		}
		namespace d53
		{
			void cos(const double * pSrc, double * pDst, int len);
			void sin(const double * pSrc, double * pDst, int len);
			void tan(const double * pSrc, double * pDst, int len);
			void acos(const double * pSrc, double * pDst, int len);
			void asin(const double * pSrc, double * pDst, int len);
			void atan(const double * pSrc, double * pDst, int len);
		}
	}

	using namespace ipp::common;
	using namespace ipp::arithmetic;
	using namespace ipp::power;
	using namespace ipp::exp_log;
	using namespace ipp::statistical;
	
	namespace f21
	{
		using namespace ipp::arithmetic::f21;
		using namespace ipp::power::f21;
		using namespace ipp::exp_log::f21;
		using namespace ipp::trigonometric::f21;
	}

	namespace f24
	{
		using namespace ipp::arithmetic::f24;
		using namespace ipp::power::f24;
		using namespace ipp::exp_log::f24;
		using namespace ipp::trigonometric::f24;
	}
	
	namespace d50
	{
		using namespace ipp::arithmetic::d50;
		using namespace ipp::power::d50;
		using namespace ipp::exp_log::d50;
		using namespace ipp::trigonometric::d50;
	}

	namespace d53
	{
		using namespace ipp::arithmetic::d53;
		using namespace ipp::power::d53;
		using namespace ipp::exp_log::d53;
		using namespace ipp::trigonometric::d53;
	}
}

#endif

