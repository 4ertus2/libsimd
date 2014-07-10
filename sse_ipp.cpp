#include <stdint.h>
#include <ipp.h>

#include "sse_ipp.h"
#include "ipp_exception.h"

namespace ipp
{
namespace common
{
	void ipp_free(void * ptr) { ippsFree(ptr); }

	// malloc
	template <> uint8_t* malloc<uint8_t>(int len) { return ippsMalloc_8u(len); }
	template <> uint16_t* malloc<uint16_t>(int len) { return ippsMalloc_16u(len); }
	template <> uint32_t* malloc<uint32_t>(int len) { return ippsMalloc_32u(len); }
	template <> int8_t* malloc<int8_t>(int len) { return ippsMalloc_8s(len); }
	template <> int16_t* malloc<int16_t>(int len) { return ippsMalloc_16s(len); }
	template <> int32_t* malloc<int32_t>(int len) { return ippsMalloc_32s(len); }
	template <> int64_t* malloc<int64_t>(int len) { return (int64_t*)ippsMalloc_64s(len); }
	template <> float* malloc<float>(int len) { return ippsMalloc_32f(len); }
	template <> double* malloc<double>(int len) { return ippsMalloc_64f(len); }
	//
	template <> uint64_t* malloc<uint64_t>(int len) { return (uint64_t*)malloc<int64_t>(len); }

	// zero
	template <> void zero(uint8_t* pDst, int len) { STATUS_CHECK(ippsZero_8u(pDst, len)); }
	template <> void zero(int16_t* pDst, int len) { STATUS_CHECK(ippsZero_16s(pDst, len)); }
	template <> void zero(int32_t* pDst, int len) { STATUS_CHECK(ippsZero_32s(pDst, len)); }
	template <> void zero(int64_t* pDst, int len) { STATUS_CHECK(ippsZero_64s((Ipp64s*)pDst, len)); }
	template <> void zero(float* pDst, int len) { STATUS_CHECK(ippsZero_32f(pDst, len)); }
	template <> void zero(double* pDst, int len) { STATUS_CHECK(ippsZero_64f(pDst, len)); }
	//
	template <> void zero(int8_t* pDst, int len) { zero<uint8_t>((uint8_t*)pDst, len); }
	template <> void zero(uint16_t* pDst, int len) { zero<int16_t>((int16_t*)pDst, len); }
	template <> void zero(uint32_t* pDst, int len) { zero<int32_t>((int32_t*)pDst, len); }
	template <> void zero(uint64_t* pDst, int len) { zero<int64_t>((int64_t*)pDst, len); }

	// set
	template <> void set(uint8_t val, uint8_t* pDst, int len) { STATUS_CHECK(ippsSet_8u(val, pDst, len)); }
	template <> void set(int16_t val, int16_t* pDst, int len) { STATUS_CHECK(ippsSet_16s(val, pDst, len)); }
	template <> void set(int32_t val, int32_t* pDst, int len) { STATUS_CHECK(ippsSet_32s(val, pDst, len)); }
	template <> void set(int64_t val, int64_t* pDst, int len) { STATUS_CHECK(ippsSet_64s(val, (Ipp64s*)pDst, len)); }
	template <> void set(float val, float* pDst, int len) { STATUS_CHECK(ippsSet_32f(val, pDst, len)); }
	template <> void set(double val, double* pDst, int len) { STATUS_CHECK(ippsSet_64f(val, pDst, len)); }
	//
	template <> void set(int8_t val, int8_t* pDst, int len) { set<uint8_t>((uint8_t)val, (uint8_t*)pDst, len); }
	template <> void set(uint16_t val, uint16_t* pDst, int len) { set<int16_t>((int16_t)val, (int16_t*)pDst, len); }
	template <> void set(uint32_t val, uint32_t* pDst, int len) { set<int32_t>((int32_t)val, (int32_t*)pDst, len); }
	template <> void set(uint64_t val, uint64_t* pDst, int len) { set<int64_t>((int64_t)val, (int64_t*)pDst, len); }

	// copy
	template <> void copy(const uint8_t* pSrc, uint8_t* pDst, int len) { STATUS_CHECK(ippsCopy_8u(pSrc, pDst, len)); }
	template <> void copy(const int16_t* pSrc, int16_t* pDst, int len) { STATUS_CHECK(ippsCopy_16s(pSrc, pDst, len)); }
	template <> void copy(const int32_t* pSrc, int32_t* pDst, int len) { STATUS_CHECK(ippsCopy_32s(pSrc, pDst, len)); }
	template <> void copy(const int64_t* pSrc, int64_t* pDst, int len) { STATUS_CHECK(ippsCopy_64s((const Ipp64s*)pSrc, (Ipp64s*)pDst, len)); }
	template <> void copy(const float* pSrc, float* pDst, int len) { STATUS_CHECK(ippsCopy_32f(pSrc, pDst, len)); }
	template <> void copy(const double* pSrc, double* pDst, int len) { STATUS_CHECK(ippsCopy_64f(pSrc, pDst, len)); }
	//
	template <> void copy(const int8_t* pSrc, int8_t* pDst, int len) { copy<uint8_t>((const uint8_t*)pSrc, (uint8_t*)pDst, len); }
	template <> void copy(const uint16_t* pSrc, uint16_t* pDst, int len) { copy<int16_t>((const int16_t*)pSrc, (int16_t*)pDst, len); }
	template <> void copy(const uint32_t* pSrc, uint32_t* pDst, int len) { copy<int32_t>((const int32_t*)pSrc, (int32_t*)pDst, len); }
	template <> void copy(const uint64_t* pSrc, uint64_t* pDst, int len) { copy<int64_t>((const int64_t*)pSrc, (int64_t*)pDst, len); }

	// move
	template <> void move(const uint8_t* pSrc, uint8_t* pDst, int len) { STATUS_CHECK(ippsMove_8u(pSrc, pDst, len)); }
	template <> void move(const int16_t* pSrc, int16_t* pDst, int len) { STATUS_CHECK(ippsMove_16s(pSrc, pDst, len)); }
	template <> void move(const int32_t* pSrc, int32_t* pDst, int len) { STATUS_CHECK(ippsMove_32s(pSrc, pDst, len)); }
	template <> void move(const int64_t* pSrc, int64_t* pDst, int len) { STATUS_CHECK(ippsMove_64s((const Ipp64s*)pSrc, (Ipp64s*)pDst, len)); }
	template <> void move(const float* pSrc, float* pDst, int len) { STATUS_CHECK(ippsMove_32f(pSrc, pDst, len)); }
	template <> void move(const double* pSrc, double* pDst, int len) { STATUS_CHECK(ippsMove_64f(pSrc, pDst, len)); }
	//
	template <> void move(const int8_t* pSrc, int8_t* pDst, int len) { move<uint8_t>((const uint8_t*)pSrc, (uint8_t*)pDst, len); }
	template <> void move(const uint16_t* pSrc, uint16_t* pDst, int len) { move<int16_t>((const int16_t*)pSrc, (int16_t*)pDst, len); }
	template <> void move(const uint32_t* pSrc, uint32_t* pDst, int len) { move<int32_t>((const int32_t*)pSrc, (int32_t*)pDst, len); }
	template <> void move(const uint64_t* pSrc, uint64_t* pDst, int len) { move<int64_t>((const int64_t*)pSrc, (int64_t*)pDst, len); }

	// convert
	template <>	void convert(const int8_t* pSrc, int16_t* pDst, int len) { STATUS_CHECK(
		ippsConvert_8s16s(pSrc, pDst, len) ); }
	template <>	void convert(const int8_t* pSrc, float* pDst, int len) { STATUS_CHECK(
		ippsConvert_8s32f(pSrc, pDst, len) ); }
	template <>	void convert(const uint8_t* pSrc, float* pDst, int len) { STATUS_CHECK(
		ippsConvert_8u32f(pSrc, pDst, len) ); }
	template <>	void convert(const int16_t* pSrc, int32_t* pDst, int len) { STATUS_CHECK(
		ippsConvert_16s32s(pSrc, pDst, len) ); }
	template <>	void convert(const int16_t* pSrc, float* pDst, int len) { STATUS_CHECK(
		ippsConvert_16s32f(pSrc, pDst, len) ); }
	template <>	void convert(const uint16_t* pSrc, float* pDst, int len) { STATUS_CHECK(
		ippsConvert_16u32f(pSrc, pDst, len) ); }
	template <>	void convert(const int32_t* pSrc, int16_t* pDst, int len) { STATUS_CHECK(
		ippsConvert_32s16s(pSrc, pDst, len) ); }
	template <>	void convert(const int32_t* pSrc, float* pDst, int len) { STATUS_CHECK(
		ippsConvert_32s32f(pSrc, pDst, len) ); }
	template <>	void convert(const int32_t* pSrc, double* pDst, int len) { STATUS_CHECK(
		ippsConvert_32s64f(pSrc, pDst, len) ); }
	template <>	void convert(const float* pSrc, double* pDst, int len) { STATUS_CHECK(
		ippsConvert_32f64f(pSrc, pDst, len) ); }
	template <>	void convert(const int64_t* pSrc, double* pDst, int len) { STATUS_CHECK(
		ippsConvert_64s64f((Ipp64s*)pSrc, pDst, len) ); }
	template <>	void convert(const double* pSrc, float* pDst, int len) { STATUS_CHECK(
		ippsConvert_64f32f(pSrc, pDst, len) ); }
}

namespace arithmetic
{
	// add
	template <> void add(const float* pSrc1, const float* pSrc2, float* pDst, int len) { STATUS_CHECK(
		ippsAdd_32f(pSrc1, pSrc2, pDst, len)); }
	// ...

	// sub
	template <> void subC(const float * pSrc, float val, float* pDst, int len) { STATUS_CHECK(
		ippsSubC_32f(pSrc, val, pDst, len)); }
	// ...

	// mul
	template <> void mulC(const float* pSrc, float val, float* pDst, int len) { STATUS_CHECK(
		ippsMulC_32f(pSrc, val, pDst, len)); }
	template <> void mul(const float* pSrc1, const float* pSrc2, float* pDst, int len) { STATUS_CHECK(
		ippsMul_32f(pSrc1, pSrc2, pDst, len)); }
	// ...

	// div
	template <> void divC(const float* pSrc, float val, float* pDst, int len) { STATUS_CHECK(
		ippsDivC_32f(pSrc, val, pDst, len)); }
	template <> void divCRev(const float* pSrc, float val, float* pDst, int len) { STATUS_CHECK(
		ippsDivCRev_32f(pSrc, val, pDst, len)); }
	// ...

	// abs
	template <> void abs(const float* pSrc, float* pDst, int len) { STATUS_CHECK(
		ippsAbs_32f(pSrc, pDst, len) ); }
	// ...
}

namespace power
{
	// sqr
	template <> void sqr(const float* pSrc, float* pDst, int len) { STATUS_CHECK(
		ippsSqr_32f(pSrc, pDst, len) ); }
	// ...

	// sqrt
	template <> void sqrt(const float* pSrc, float* pDst, int len) { STATUS_CHECK(
		ippsSqrt_32f(pSrc, pDst, len)); }
	// ...

	// cubrt
	template <> void cubrt(const float* pSrc, float* pDst, int len) { STATUS_CHECK(
		ippsCubrt_32f(pSrc, pDst, len) ); }
	// ...
}

namespace exp_log
{
	// exp
	template <> void exp(const float* pSrc, float* pDst, int len) { STATUS_CHECK(
		ippsExp_32f(pSrc, pDst, len)); }
	// ...

	// ln
	template <> void ln(const float* pSrc, float* pDst, int len) { STATUS_CHECK(
		ippsLn_32f(pSrc, pDst, len)); }
	template <> void ln(const double* pSrc, double* pDst, int len) { STATUS_CHECK(
		ippsLn_64f(pSrc, pDst, len)); }
	// ...
}

namespace threshold
{
	template <> void threshold_LT(const float* pSrc, float* pDst, int len, float level) { STATUS_CHECK(
		ippsThreshold_LT_32f(pSrc, pDst, len, level)); }
	template <> void threshold_GT(const float* pSrc, float* pDst, int len, float level) { STATUS_CHECK(
		ippsThreshold_GT_32f(pSrc, pDst, len, level)); }
	template <> void threshold_LTAbs(const float* pSrc, float* pDst, int len, float level) { STATUS_CHECK(
		ippsThreshold_LTAbs_32f(pSrc, pDst, len, level)); }
	template <> void threshold_GTAbs(const float* pSrc, float* pDst, int len, float level) { STATUS_CHECK(
		ippsThreshold_GTAbs_32f(pSrc, pDst, len, level)); }
}

namespace statistical
{
	// sum
	template <> void sum(const float* pSrc, int len, float* pSum) { STATUS_CHECK(
		ippsSum_32f(pSrc, len, pSum, ippAlgHintNone)); }
	// ...

	// mean
	// ...
	template <> void mean(const float* pSrc, int len, float* pMean) { STATUS_CHECK(
		ippsMean_32f(pSrc, len, pMean, ippAlgHintNone)); }
	// ...

	// max
	template <> void max(const float* pSrc, int len, float* pMax) { STATUS_CHECK(
		ippsMax_32f(pSrc, len, pMax)); }
	template <> void maxIndx(const float* pSrc, int len, float* pMax, int* pIndx) { STATUS_CHECK(
		ippsMaxIndx_32f(pSrc, len, pMax, pIndx)); }
	// ...

	// min
	template <> void min(const float* pSrc, int len, float* pMin) { STATUS_CHECK(
		ippsMin_32f(pSrc, len, pMin) ); }
	// ...

	// minMax
	template <> void minMax(const float* pSrc, int len, float* pMin, float* pMax) { STATUS_CHECK(
		ippsMinMax_32f(pSrc, len, pMin, pMax)); }
	// ...
}

// TODO: rounding

} // ipp


namespace ipp_f24_d53
{
namespace arithmetic
{
	// add
	template <> void add(const float* pSrc1, const float* pSrc2, float* pDst, int len) { STATUS_CHECK(
		ippsAdd_32f_A24(pSrc1, pSrc2, pDst, len)); }
	template <> void add(const double* pSrc1, const double* pSrc2, double* pDst, int len) { STATUS_CHECK(
		ippsAdd_64f_A53(pSrc1, pSrc2, pDst, len)); }

	// sub
	template <> void sub(const float* pSrc1, const float* pSrc2, float* pDst, int len) { STATUS_CHECK(
		ippsSub_32f_A24(pSrc1, pSrc2, pDst, len)); }
	template <> void sub(const double* pSrc1, const double* pSrc2, double* pDst, int len) { STATUS_CHECK(
		ippsSub_64f_A53(pSrc1, pSrc2, pDst, len)); }

	// mul
	template <> void mul(const float* pSrc1, const float* pSrc2, float* pDst, int len) { STATUS_CHECK(
		ippsMul_32f_A24(pSrc1, pSrc2, pDst, len)); }
	template <> void mul(const double* pSrc1, const double* pSrc2, double* pDst, int len) { STATUS_CHECK(
		ippsMul_64f_A53(pSrc1, pSrc2, pDst, len)); }

	// abs
	template <> void abs(const float* pSrc, float* pDst, int len) { STATUS_CHECK(
		ippsAbs_32f_A24(pSrc, pDst, len)); }
	template <> void abs(const double* pSrc, double* pDst, int len) { STATUS_CHECK(
		ippsAbs_64f_A53(pSrc, pDst, len)); }
}

namespace power
{
	// sqr
	template <> void sqr(const float* pSrc, float* pDst, int len) { STATUS_CHECK(
		ippsSqr_32f_A24(pSrc, pDst, len)); }
	template <> void sqr(const double* pSrc, double* pDst, int len) { STATUS_CHECK(
		ippsSqr_64f_A53(pSrc, pDst, len)); }
}

namespace trigonometric
{
	// cos
	template <> void cos(const float* pSrc, float* pDst, int len) { STATUS_CHECK(
		ippsCos_32f_A24(pSrc, pDst, len)); }
	template <> void cos(const double* pSrc, double* pDst, int len) { STATUS_CHECK(
		ippsCos_64f_A53(pSrc, pDst, len)); }

	// sin
	template <> void sin(const float* pSrc, float* pDst, int len) { STATUS_CHECK(
		ippsSin_32f_A24(pSrc, pDst, len)); }
	template <> void sin(const double* pSrc, double* pDst, int len) { STATUS_CHECK(
		ippsSin_64f_A53(pSrc, pDst, len)); }
}

	// TODO

} // ipp_f24_d53


// TODO: ipp_f21_d50
// TODO: ipp_f11_d26

