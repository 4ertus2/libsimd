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
	// addC
	template <> void addC(const float * pSrc, float val, float * pDst, int len) { STATUS_CHECK(
		ippsAddC_32f(pSrc, val, pDst, len)); }
	template <> void addC(const double * pSrc, double val, double * pDst, int len) { STATUS_CHECK(
		ippsAddC_64f(pSrc, val, pDst, len)); }
	template <> void addC(const uint8_t * pSrc, uint8_t val, uint8_t * pDst, int len) { STATUS_CHECK(
		ippsAddC_8u_Sfs(pSrc, val, pDst, len, 0)); }
	template <> void addC(const uint16_t * pSrc, uint16_t val, uint16_t * pDst, int len) { STATUS_CHECK(
		ippsAddC_16u_Sfs(pSrc, val, pDst, len, 0)); }
	template <> void addC(const int16_t * pSrc, int16_t val, int16_t * pDst, int len) {STATUS_CHECK(
		ippsAddC_16s_Sfs(pSrc, val, pDst, len, 0)); }
	template <> void addC(const int32_t * pSrc, int32_t val, int32_t * pDst, int len) { STATUS_CHECK(
		ippsAddC_32s_Sfs(pSrc, val, pDst, len, 0)); }

	// add
	template <> void add(const float * pSrc1, const float * pSrc2, float * pDst, int len) { STATUS_CHECK(
		ippsAdd_32f(pSrc1, pSrc2, pDst, len)); }
	template <> void add(const double * pSrc1, const double * pSrc2, double * pDst, int len) { STATUS_CHECK(
		ippsAdd_64f(pSrc1, pSrc2, pDst, len)); }
	template <> void add(const uint8_t * pSrc1, const uint8_t * pSrc2, uint8_t * pDst, int len) { STATUS_CHECK(
		ippsAdd_8u_Sfs(pSrc1, pSrc2, pDst, len, 0)); }
	template <> void add(const uint16_t * pSrc1, const uint16_t * pSrc2, uint16_t * pDst, int len) { STATUS_CHECK(
		ippsAdd_16u(pSrc1, pSrc2, pDst, len)); }
	template <> void add(const int16_t * pSrc1, const int16_t * pSrc2, int16_t * pDst, int len) { STATUS_CHECK(
		ippsAdd_16s(pSrc1, pSrc2, pDst, len)); }
	template <> void add(const uint32_t * pSrc1, const uint32_t * pSrc2, uint32_t * pDst, int len) { STATUS_CHECK(
		ippsAdd_32u(pSrc1, pSrc2, pDst, len)); }
	template <> void add(const int32_t * pSrc1, const int32_t * pSrc2, int32_t * pDst, int len) { STATUS_CHECK(
		ippsAdd_32s_Sfs(pSrc1, pSrc2, pDst, len, 0)); }
	template <> void add(const int64_t * pSrc1, const int64_t * pSrc2, int64_t * pDst, int len) { STATUS_CHECK(
		ippsAdd_64s_Sfs((const Ipp64s*)pSrc1, (const Ipp64s*)pSrc2, (Ipp64s*)pDst, len, 0)); }
	namespace f24
	{
		void add(const float * pSrc1, const float * pSrc2, float * pDst, int len) { STATUS_CHECK(
			ippsAdd_32f_A24(pSrc1, pSrc2, pDst, len)); }
	}
	namespace d53
	{
		void add(const double * pSrc1, const double * pSrc2, double * pDst, int len) { STATUS_CHECK(
			ippsAdd_64f_A53(pSrc1, pSrc2, pDst, len)); }
	}

	// subC
	template <> void subC(const float * pSrc, float val, float * pDst, int len) { STATUS_CHECK(
		ippsSubC_32f(pSrc, val, pDst, len)); }
	template <> void subC(const double * pSrc, double val, double * pDst, int len) { STATUS_CHECK(
		ippsSubC_64f(pSrc, val, pDst, len)); }
	template <> void subC(const uint8_t * pSrc, uint8_t val, uint8_t * pDst, int len) { STATUS_CHECK(
		ippsSubC_8u_Sfs(pSrc, val, pDst, len, 0)); }
	template <> void subC(const uint16_t * pSrc, uint16_t val, uint16_t * pDst, int len) { STATUS_CHECK(
		ippsSubC_16u_Sfs(pSrc, val, pDst, len, 0)); }
	template <> void subC(const int16_t * pSrc, int16_t val, int16_t * pDst, int len) { STATUS_CHECK(
		ippsSubC_16s_Sfs(pSrc, val, pDst, len, 0)); }
	template <> void subC(const int32_t * pSrc, int32_t val, int32_t * pDst, int len) { STATUS_CHECK(
		ippsSubC_32s_Sfs(pSrc, val, pDst, len, 0)); }

	// subCRev
	template <> void subCRev(const float * pSrc, float val, float * pDst, int len) { STATUS_CHECK(
		ippsSubCRev_32f(pSrc, val, pDst, len)); }
	template <> void subCRev(const double * pSrc, double val, double * pDst, int len) { STATUS_CHECK(
		ippsSubCRev_64f(pSrc, val, pDst, len)); }
	template <> void subCRev(const uint8_t * pSrc, uint8_t val, uint8_t * pDst, int len) { STATUS_CHECK(
		ippsSubCRev_8u_Sfs(pSrc, val, pDst, len, 0)); }
	template <> void subCRev(const uint16_t * pSrc, uint16_t val, uint16_t * pDst, int len) { STATUS_CHECK(
		ippsSubCRev_16u_Sfs(pSrc, val, pDst, len, 0)); }
	template <> void subCRev(const int16_t * pSrc, int16_t val, int16_t * pDst, int len) { STATUS_CHECK(
		ippsSubCRev_16s_Sfs(pSrc, val, pDst, len, 0)); }
	template <> void subCRev(const int32_t * pSrc, int32_t val, int32_t * pDst, int len) { STATUS_CHECK(
		ippsSubCRev_32s_Sfs(pSrc, val, pDst, len, 0)); }

	// sub (fixed parameters order)
	template <> void sub(const float * pSrc1, const float * pSrc2, float * pDst, int len) { STATUS_CHECK(
		ippsSub_32f(pSrc2, pSrc1, pDst, len)); }
	template <> void sub(const double * pSrc1, const double * pSrc2, double * pDst, int len) { STATUS_CHECK(
		ippsSub_64f(pSrc2, pSrc1, pDst, len)); }
	template <> void sub(const int16_t * pSrc1, const int16_t * pSrc2, int16_t * pDst, int len) { STATUS_CHECK(
		ippsSub_16s(pSrc2, pSrc1, pDst, len)); }
	template <> void sub(const uint8_t * pSrc1, const uint8_t * pSrc2, uint8_t * pDst, int len) { STATUS_CHECK(
		ippsSub_8u_Sfs(pSrc2, pSrc1, pDst, len, 0)); }
	template <> void sub(const uint16_t * pSrc1, const uint16_t * pSrc2, uint16_t * pDst, int len) { STATUS_CHECK(
		ippsSub_16u_Sfs(pSrc2, pSrc1, pDst, len, 0)); }
	template <> void sub(const int32_t * pSrc1, const int32_t * pSrc2, int32_t * pDst, int len) { STATUS_CHECK(
		ippsSub_32s_Sfs(pSrc2, pSrc1, pDst, len, 0)); }
	namespace f24
	{
		void sub(const float * pSrc1, const float * pSrc2, float * pDst, int len) { STATUS_CHECK(
			ippsSub_32f_A24(pSrc1, pSrc2, pDst, len)); }
	}
	namespace d53
	{
		void sub(const double * pSrc1, const double * pSrc2, double * pDst, int len) { STATUS_CHECK(
			ippsSub_64f_A53(pSrc1, pSrc2, pDst, len)); }
	}

	// mulC
	template <> void mulC(const float * pSrc, float val, float * pDst, int len) { STATUS_CHECK(
		ippsMulC_32f(pSrc, val, pDst, len)); }
	template <> void mulC(const double * pSrc, double val, double * pDst, int len) { STATUS_CHECK(
		ippsMulC_64f(pSrc, val, pDst, len)); }
	template <> void mulC(const uint8_t * pSrc, uint8_t val, uint8_t * pDst, int len) { STATUS_CHECK(
		ippsMulC_8u_Sfs(pSrc, val, pDst, len, 0)); }
	template <> void mulC(const uint16_t * pSrc, uint16_t val, uint16_t * pDst, int len) { STATUS_CHECK(
		ippsMulC_16u_Sfs(pSrc, val, pDst, len, 0)); }
	template <> void mulC(const int16_t * pSrc, int16_t val, int16_t * pDst, int len) { STATUS_CHECK(
		ippsMulC_16s_Sfs(pSrc, val, pDst, len, 0)); }
	template <> void mulC(const int32_t * pSrc, int32_t val, int32_t * pDst, int len) { STATUS_CHECK(
		ippsMulC_32s_Sfs(pSrc, val, pDst, len, 0)); }

	// mul
	template <> void mul(const float * pSrc1, const float * pSrc2, float * pDst, int len) { STATUS_CHECK(
		ippsMul_32f(pSrc1, pSrc2, pDst, len)); }
	template <> void mul(const double * pSrc1, const double * pSrc2, double * pDst, int len) { STATUS_CHECK(
		ippsMul_64f(pSrc1, pSrc2, pDst, len)); }
	template <> void mul(const int16_t * pSrc1, const int16_t * pSrc2, int16_t * pDst, int len) { STATUS_CHECK(
		ippsMul_16s(pSrc1, pSrc2, pDst, len)); }
	template <> void mul(const uint8_t * pSrc1, const uint8_t * pSrc2, uint8_t * pDst, int len) { STATUS_CHECK(
		ippsMul_8u_Sfs(pSrc1, pSrc2, pDst, len, 0)); }
	template <> void mul(const uint16_t * pSrc1, const uint16_t * pSrc2, uint16_t * pDst, int len) { STATUS_CHECK(
		ippsMul_16u_Sfs(pSrc1, pSrc2, pDst, len, 0)); }
	template <> void mul(const int32_t * pSrc1, const int32_t * pSrc2, int32_t * pDst, int len) { STATUS_CHECK(
		ippsMul_32s_Sfs(pSrc1, pSrc2, pDst, len, 0)); }
	namespace f24
	{
		void mul(const float* pSrc1, const float* pSrc2, float* pDst, int len) { STATUS_CHECK(
			ippsMul_32f_A24(pSrc1, pSrc2, pDst, len)); }
	}
	namespace d53
	{
		void mul(const double* pSrc1, const double* pSrc2, double* pDst, int len) { STATUS_CHECK(
			ippsMul_64f_A53(pSrc1, pSrc2, pDst, len)); }
	}

	// divC
	template <> void divC(const float * pSrc, float val, float * pDst, int len) { STATUS_CHECK(
		ippsDivC_32f(pSrc, val, pDst, len)); }
	template <> void divC(const double * pSrc, double val, double * pDst, int len) { STATUS_CHECK(
		ippsDivC_64f(pSrc, val, pDst, len)); }
	template <> void divC(const uint8_t * pSrc, uint8_t val, uint8_t * pDst, int len) { STATUS_CHECK(
		ippsDivC_8u_Sfs(pSrc, val, pDst, len, 0)); }
	template <> void divC(const uint16_t * pSrc, uint16_t val, uint16_t * pDst, int len) { STATUS_CHECK(
		ippsDivC_16u_Sfs(pSrc, val, pDst, len, 0)); }
	template <> void divC(const int16_t * pSrc, int16_t val, int16_t * pDst, int len) { STATUS_CHECK(
		ippsDivC_16s_Sfs(pSrc, val, pDst, len, 0)); }

	// divCRev
	template <> void divCRev(const float * pSrc, float val, float * pDst, int len) { STATUS_CHECK(
		ippsDivCRev_32f(pSrc, val, pDst, len)); }

	// div (fixed parameters order)
	template <> void div(const float * pSrc1, const float * pSrc2, float * pDst, int len) { STATUS_CHECK(
		ippsDiv_32f(pSrc2, pSrc1, pDst, len)); }
	template <> void div(const double * pSrc1, const double * pSrc2, double * pDst, int len) { STATUS_CHECK(
		ippsDiv_64f(pSrc2, pSrc1, pDst, len)); }
	template <> void div(const uint8_t * pSrc1, const uint8_t * pSrc2, uint8_t * pDst, int len) { STATUS_CHECK(
		ippsDiv_8u_Sfs(pSrc2, pSrc1, pDst, len, 0)); }
	template <> void div(const uint16_t * pSrc1, const uint16_t * pSrc2, uint16_t * pDst, int len) { STATUS_CHECK(
		ippsDiv_16u_Sfs(pSrc2, pSrc1, pDst, len, 0)); }
	template <> void div(const int16_t * pSrc1, const int16_t * pSrc2, int16_t * pDst, int len) { STATUS_CHECK(
		ippsDiv_16s_Sfs(pSrc2, pSrc1, pDst, len, 0)); }
	template <> void div(const int32_t * pSrc1, const int32_t * pSrc2, int32_t * pDst, int len) { STATUS_CHECK(
		ippsDiv_32s_Sfs(pSrc2, pSrc1, pDst, len, 0)); }
	namespace f21
	{
		void div(const float * pSrc1, const float * pSrc2, float * pDst, int len) { STATUS_CHECK(
			ippsDiv_32f_A21(pSrc1, pSrc2, pDst, len)); }
	}
	namespace f24
	{
		void div(const float * pSrc1, const float * pSrc2, float * pDst, int len) { STATUS_CHECK(
			ippsDiv_32f_A24(pSrc1, pSrc2, pDst, len)); }
	}
	namespace d50
	{
		void div(const double * pSrc1, const double * pSrc2, double * pDst, int len) { STATUS_CHECK(
			ippsDiv_64f_A50(pSrc1, pSrc2, pDst, len)); }
	}
	namespace d53
	{
		void div(const double * pSrc1, const double * pSrc2, double * pDst, int len) { STATUS_CHECK(
			ippsDiv_64f_A53(pSrc1, pSrc2, pDst, len)); }
	}

	// abs
	template <> void abs(const float * pSrc, float * pDst, int len) { STATUS_CHECK(ippsAbs_32f(pSrc, pDst, len) ); }
	template <> void abs(const double * pSrc, double * pDst, int len) { STATUS_CHECK(ippsAbs_64f(pSrc, pDst, len) ); }
	template <> void abs(const int16_t * pSrc, int16_t * pDst, int len) { STATUS_CHECK(ippsAbs_16s(pSrc, pDst, len) ); }
	template <> void abs(const int32_t * pSrc, int32_t * pDst, int len) { STATUS_CHECK(ippsAbs_32s(pSrc, pDst, len) ); }
	//
	template <> void abs(const uint8_t * pSrc, uint8_t * pDst, int len) { copy(pSrc, pDst, len); }
	template <> void abs(const uint16_t * pSrc, uint16_t * pDst, int len) { copy(pSrc, pDst, len); }
	template <> void abs(const uint32_t * pSrc, uint32_t * pDst, int len) { copy(pSrc, pDst, len); }
	namespace f24
	{
		void abs(const float* pSrc, float* pDst, int len) { STATUS_CHECK(ippsAbs_32f_A24(pSrc, pDst, len)); }
	}
	namespace d53
	{
		void abs(const double* pSrc, double* pDst, int len) { STATUS_CHECK(ippsAbs_64f_A53(pSrc, pDst, len)); }
	}
}

namespace power
{
	// inv
	template <> void inv(const float * pSrc, float * pDst, int len) { STATUS_CHECK(ippsDivCRev_32f(pSrc, 1.0f, pDst, len)); }
	template <> void inv(const uint16_t * pSrc, uint16_t * pDst, int len) { STATUS_CHECK(ippsDivCRev_16u(pSrc, 1, pDst, len)); }
	namespace f21
	{
		void inv(const float * pSrc, float * pDst, int len) { STATUS_CHECK(ippsInv_32f_A21(pSrc, pDst, len)); }
	}
	namespace f24
	{
		void inv(const float * pSrc, float * pDst, int len) { STATUS_CHECK(ippsInv_32f_A24(pSrc, pDst, len)); }
	}
	namespace d50
	{
		void inv(const double * pSrc, double * pDst, int len) { STATUS_CHECK(ippsInv_64f_A50(pSrc, pDst, len)); }
	}
	namespace d53
	{
		void inv(const double * pSrc, double * pDst, int len) { STATUS_CHECK(ippsInv_64f_A53(pSrc, pDst, len)); }
	}

	// sqr
	template <> void sqr(const float * pSrc, float * pDst, int len) { STATUS_CHECK(
		ippsSqr_32f(pSrc, pDst, len) ); }
	template <> void sqr(const double * pSrc, double * pDst, int len) { STATUS_CHECK(
		ippsSqr_64f(pSrc, pDst, len) ); }
	template <> void sqr(const uint8_t * pSrc, uint8_t * pDst, int len) { STATUS_CHECK(
		ippsSqr_8u_Sfs(pSrc, pDst, len, 0) ); }
	template <> void sqr(const uint16_t * pSrc, uint16_t * pDst, int len) { STATUS_CHECK(
		ippsSqr_16u_Sfs(pSrc, pDst, len, 0) ); }
	template <> void sqr(const int16_t * pSrc, int16_t * pDst, int len) { STATUS_CHECK(
		ippsSqr_16s_Sfs(pSrc, pDst, len, 0) ); }
	namespace f24
	{
		void sqr(const float * pSrc, float * pDst, int len) { STATUS_CHECK(ippsSqr_32f_A24(pSrc, pDst, len)); }
	}
	namespace d53
	{
		void sqr(const double * pSrc, double * pDst, int len) { STATUS_CHECK(ippsSqr_64f_A53(pSrc, pDst, len)); }
	}

	// sqrt
	template <> void sqrt(const float * pSrc, float * pDst, int len) { STATUS_CHECK(
		ippsSqrt_32f(pSrc, pDst, len)); }
	template <> void sqrt(const double * pSrc, double * pDst, int len) { STATUS_CHECK(
		ippsSqrt_64f(pSrc, pDst, len)); }
	template <> void sqrt(const uint8_t * pSrc, uint8_t * pDst, int len) { STATUS_CHECK(
		ippsSqrt_8u_Sfs(pSrc, pDst, len, 0)); }
	template <> void sqrt(const uint16_t * pSrc, uint16_t * pDst, int len) { STATUS_CHECK(
		ippsSqrt_16u_Sfs(pSrc, pDst, len, 0)); }
	template <> void sqrt(const int16_t * pSrc, int16_t * pDst, int len) { STATUS_CHECK(
		ippsSqrt_16s_Sfs(pSrc, pDst, len, 0)); }
	namespace f21
	{
		void sqrt(const float * pSrc, float * pDst, int len) { STATUS_CHECK(ippsSqrt_32f_A21(pSrc, pDst, len)); }
	}
	namespace f24
	{
		void sqrt(const float * pSrc, float * pDst, int len) { STATUS_CHECK(ippsSqrt_32f_A24(pSrc, pDst, len)); }
	}
	namespace d50
	{
		void sqrt(const double * pSrc, double * pDst, int len) { STATUS_CHECK(ippsSqrt_64f_A50(pSrc, pDst, len)); }
	}
	namespace d53
	{
		void sqrt(const double * pSrc, double * pDst, int len) { STATUS_CHECK(ippsSqrt_64f_A53(pSrc, pDst, len)); }
	}

	// invSqrt

	// powx
	namespace f21
	{
		void powx(const float * pSrc, const float constValue, float * pDst, int len) { STATUS_CHECK(
			ippsPowx_32f_A21(pSrc, constValue, pDst, len)); }
	}
	namespace f24
	{
		void powx(const float * pSrc, const float constValue, float * pDst, int len) { STATUS_CHECK(
			ippsPowx_32f_A24(pSrc, constValue, pDst, len)); }
	}
	namespace d50
	{
		void powx(const double * pSrc, const double constValue, double * pDst, int len) { STATUS_CHECK(
			ippsPowx_64f_A50(pSrc, constValue, pDst, len)); }
	}
	namespace d53
	{
		void powx(const double * pSrc, const double constValue, double * pDst, int len) { STATUS_CHECK(
			ippsPowx_64f_A53(pSrc, constValue, pDst, len)); }
	}

	// pow
	namespace f21
	{
		void pow(const float * pSrc1, const float * pSrc2, float * pDst, int len) { STATUS_CHECK(
			ippsPow_32f_A21(pSrc1, pSrc2, pDst, len)); }
	}
	namespace f24
	{
		void pow(const float * pSrc1, const float * pSrc2, float * pDst, int len) { STATUS_CHECK(
			ippsPow_32f_A24(pSrc1, pSrc2, pDst, len)); }
	}
	namespace d50
	{
		void pow(const double * pSrc1, const double * pSrc2, double * pDst, int len) { STATUS_CHECK(
			ippsPow_64f_A50(pSrc1, pSrc2, pDst, len)); }
	}
	namespace d53
	{
		void pow(const double * pSrc1, const double * pSrc2, double * pDst, int len) { STATUS_CHECK(
			ippsPow_64f_A53(pSrc1, pSrc2, pDst, len)); }
	}
}

namespace exp_log
{
	// exp
	template <> void exp(const float * pSrc, float * pDst, int len) { STATUS_CHECK(
		ippsExp_32f(pSrc, pDst, len)); }
	template <> void exp(const double * pSrc, double * pDst, int len) { STATUS_CHECK(
		ippsExp_64f(pSrc, pDst, len)); }	
	template <> void exp(const int16_t * pSrc, int16_t * pDst, int len) { STATUS_CHECK(
		ippsExp_16s_Sfs(pSrc, pDst, len, 0)); }
	template <> void exp(const int32_t * pSrc, int32_t * pDst, int len) { STATUS_CHECK(
		ippsExp_32s_Sfs(pSrc, pDst, len, 0)); }
	template <> void exp(const int64_t * pSrc, int64_t * pDst, int len) { STATUS_CHECK(
		ippsExp_64s_Sfs((const Ipp64s *)pSrc, (Ipp64s *)pDst, len, 0)); }

	// ln
	template <> void ln(const float * pSrc, float * pDst, int len) { STATUS_CHECK(
		ippsLn_32f(pSrc, pDst, len)); }
	template <> void ln(const double * pSrc, double * pDst, int len) { STATUS_CHECK(
		ippsLn_64f(pSrc, pDst, len)); }
	template <> void ln(const int16_t * pSrc, int16_t * pDst, int len) { STATUS_CHECK(
		ippsLn_16s_Sfs(pSrc, pDst, len, 0)); }
	template <> void ln(const int32_t * pSrc, int32_t * pDst, int len) { STATUS_CHECK(
		ippsLn_32s_Sfs(pSrc, pDst, len, 0)); }
}

namespace statistical
{
	// max
	template <> void max(const float * pSrc, int len, float * pMax) { STATUS_CHECK(
		ippsMax_32f(pSrc, len, pMax)); }
	template <> void max(const double * pSrc, int len, double * pMax) { STATUS_CHECK(
		ippsMax_64f(pSrc, len, pMax)); }
	template <> void max(const int16_t * pSrc, int len, int16_t * pMax) { STATUS_CHECK(
		ippsMax_16s(pSrc, len, pMax)); }
	template <> void max(const int32_t * pSrc, int len, int32_t * pMax) { STATUS_CHECK(
		ippsMax_32s(pSrc, len, pMax)); }

	// maxIndx
	template <> void maxIndx(const float * pSrc, int len, float * pMax, int * pIndx) { STATUS_CHECK(
		ippsMaxIndx_32f(pSrc, len, pMax, pIndx)); }
	template <> void maxIndx(const double * pSrc, int len, double * pMax, int * pIndx) { STATUS_CHECK(
		ippsMaxIndx_64f(pSrc, len, pMax, pIndx)); }
	template <> void maxIndx(const int16_t * pSrc, int len, int16_t * pMax, int * pIndx) { STATUS_CHECK(
		ippsMaxIndx_16s(pSrc, len, pMax, pIndx)); }
	template <> void maxIndx(const int32_t * pSrc, int len, int32_t * pMax, int * pIndx) { STATUS_CHECK(
		ippsMaxIndx_32s(pSrc, len, pMax, pIndx)); }

	// min
	template <> void min(const float * pSrc, int len, float * pMin) { STATUS_CHECK(
		ippsMin_32f(pSrc, len, pMin) ); }
	template <> void min(const double * pSrc, int len, double * pMin) { STATUS_CHECK(
		ippsMin_64f(pSrc, len, pMin) ); }
	template <> void min(const int16_t * pSrc, int len, int16_t * pMin) { STATUS_CHECK(
		ippsMin_16s(pSrc, len, pMin) ); }
	template <> void min(const int32_t * pSrc, int len, int32_t * pMin) { STATUS_CHECK(
		ippsMin_32s(pSrc, len, pMin) ); }

	// minIndx
	template <> void minIndx(const float * pSrc, int len, float * pMax, int * pIndx) { STATUS_CHECK(
		ippsMinIndx_32f(pSrc, len, pMax, pIndx)); }
	template <> void minIndx(const double * pSrc, int len, double * pMax, int * pIndx) { STATUS_CHECK(
		ippsMinIndx_64f(pSrc, len, pMax, pIndx)); }
	template <> void minIndx(const int16_t * pSrc, int len, int16_t * pMax, int * pIndx) { STATUS_CHECK(
		ippsMinIndx_16s(pSrc, len, pMax, pIndx)); }
	template <> void minIndx(const int32_t * pSrc, int len, int32_t * pMax, int * pIndx) { STATUS_CHECK(
		ippsMinIndx_32s(pSrc, len, pMax, pIndx)); }

	// minMax
	template <> void minMax(const float * pSrc, int len, float * pMin, float * pMax) { STATUS_CHECK(
		ippsMinMax_32f(pSrc, len, pMin, pMax)); }
	template <> void minMax(const double * pSrc, int len, double * pMin, double * pMax) { STATUS_CHECK(
		ippsMinMax_64f(pSrc, len, pMin, pMax)); }
	template <> void minMax(const uint8_t * pSrc, int len, uint8_t * pMin, uint8_t * pMax) { STATUS_CHECK(
		ippsMinMax_8u(pSrc, len, pMin, pMax)); }
	template <> void minMax(const uint16_t * pSrc, int len, uint16_t * pMin, uint16_t * pMax) { STATUS_CHECK(
		ippsMinMax_16u(pSrc, len, pMin, pMax)); }
	template <> void minMax(const uint32_t * pSrc, int len, uint32_t * pMin, uint32_t * pMax) { STATUS_CHECK(
		ippsMinMax_32u(pSrc, len, pMin, pMax)); }
	template <> void minMax(const int16_t * pSrc, int len, int16_t * pMin, int16_t * pMax) { STATUS_CHECK(
		ippsMinMax_16s(pSrc, len, pMin, pMax)); }
	template <> void minMax(const int32_t * pSrc, int len, int32_t * pMin, int32_t * pMax) { STATUS_CHECK(
		ippsMinMax_32s(pSrc, len, pMin, pMax)); }

	// minMaxIndx
	template <> void minMaxIndx(const float * pSrc, int len, float * pMin, int * pMinIndx, float * pMax, int * pMaxIndx) {
		STATUS_CHECK(ippsMinMaxIndx_32f(pSrc, len, pMin, pMinIndx, pMax, pMaxIndx)); }
	template <> void minMaxIndx(const double * pSrc, int len, double * pMin, int * pMinIndx, double * pMax, int * pMaxIndx) {
		STATUS_CHECK(ippsMinMaxIndx_64f(pSrc, len, pMin, pMinIndx, pMax, pMaxIndx)); }
	template <> void minMaxIndx(const uint8_t * pSrc, int len, uint8_t * pMin, int * pMinIndx, uint8_t * pMax, int * pMaxIndx) {
		STATUS_CHECK(ippsMinMaxIndx_8u(pSrc, len, pMin, pMinIndx, pMax, pMaxIndx)); }
	template <> void minMaxIndx(const uint16_t * pSrc, int len, uint16_t * pMin, int * pMinIndx, uint16_t * pMax, int * pMaxIndx) {
		STATUS_CHECK(ippsMinMaxIndx_16u(pSrc, len, pMin, pMinIndx, pMax, pMaxIndx)); }
	template <> void minMaxIndx(const uint32_t * pSrc, int len, uint32_t * pMin, int * pMinIndx, uint32_t * pMax, int * pMaxIndx) {
		STATUS_CHECK(ippsMinMaxIndx_32u(pSrc, len, pMin, pMinIndx, pMax, pMaxIndx)); }
	template <> void minMaxIndx(const int16_t * pSrc, int len, int16_t * pMin, int * pMinIndx, int16_t * pMax, int * pMaxIndx) {
		STATUS_CHECK(ippsMinMaxIndx_16s(pSrc, len, pMin, pMinIndx, pMax, pMaxIndx)); }
	template <> void minMaxIndx(const int32_t * pSrc, int len, int32_t * pMin, int * pMinIndx, int32_t * pMax, int * pMaxIndx) {
		STATUS_CHECK(ippsMinMaxIndx_32s(pSrc, len, pMin, pMinIndx, pMax, pMaxIndx)); }

	// sum
	template <> void sum(const float * pSrc, int len, float * pSum) { STATUS_CHECK(
		ippsSum_32f(pSrc, len, pSum, ippAlgHintNone)); }
	template <> void sum(const double * pSrc, int len, double * pSum) { STATUS_CHECK(
		ippsSum_64f(pSrc, len, pSum)); }
	template <> void sum(const int16_t * pSrc, int len, int16_t * pSum) { STATUS_CHECK(
		ippsSum_16s_Sfs(pSrc, len, pSum, 0)); }
	template <> void sum(const int16_t * pSrc, int len, int32_t * pSum) { STATUS_CHECK(
		ippsSum_16s32s_Sfs(pSrc, len, pSum, 0)); }
	template <> void sum(const int32_t * pSrc, int len, int32_t * pSum) { STATUS_CHECK(
		ippsSum_32s_Sfs(pSrc, len, pSum, 0)); }

	// mean
	template <> void mean(const float * pSrc, int len, float * pMean) { STATUS_CHECK(
		ippsMean_32f(pSrc, len, pMean, ippAlgHintNone)); }
	template <> void mean(const double * pSrc, int len, double * pMean) { STATUS_CHECK(
		ippsMean_64f(pSrc, len, pMean)); }
	template <> void mean(const int16_t * pSrc, int len, int16_t * pMean) { STATUS_CHECK(
		ippsMean_16s_Sfs(pSrc, len, pMean, 0)); }
	template <> void mean(const int32_t * pSrc, int len, int32_t * pMean) { STATUS_CHECK(
		ippsMean_32s_Sfs(pSrc, len, pMean, 0)); }

	// stdDev
	template <> void stdDev(const float * pSrc, int len, float * pStdDev) { STATUS_CHECK(
		ippsStdDev_32f(pSrc, len, pStdDev, ippAlgHintNone)); }
	template <> void stdDev(const double * pSrc, int len, double * pStdDev) { STATUS_CHECK(
		ippsStdDev_64f(pSrc, len, pStdDev)); }
	template <> void stdDev(const int16_t * pSrc, int len, int16_t * pStdDev) { STATUS_CHECK(
		ippsStdDev_16s_Sfs(pSrc, len, pStdDev, 0)); }

	// meanStdDev
	template <> void meanStdDev(const float * pSrc, int len, float * pMean, float * pStdDev) { STATUS_CHECK(
		ippsMeanStdDev_32f(pSrc, len, pMean, pStdDev, ippAlgHintNone)); }
	template <> void meanStdDev(const double * pSrc, int len, double * pMean, double * pStdDev) { STATUS_CHECK(
		ippsMeanStdDev_64f(pSrc, len, pMean, pStdDev)); }
	template <> void meanStdDev(const int16_t * pSrc, int len, int16_t * pMean, int16_t * pStdDev) { STATUS_CHECK(
		ippsMeanStdDev_16s_Sfs(pSrc, len, pMean, pStdDev, 0)); }

	// normInf
	template <> void normInf(const float * pSrc, int len, float * pNorm) { STATUS_CHECK(
		ippsNorm_Inf_32f(pSrc, len, pNorm)); }
	template <> void normInf(const double * pSrc, int len, double * pNorm) { STATUS_CHECK(
		ippsNorm_Inf_64f(pSrc, len, pNorm)); }
	template <> void normInf(const int16_t * pSrc, int len, float * pNorm) { STATUS_CHECK(
		ippsNorm_Inf_16s32f(pSrc, len, pNorm)); }

	// normL1
	template <> void normL1(const float * pSrc, int len, float * pNorm) { STATUS_CHECK(
		ippsNorm_L1_32f(pSrc, len, pNorm)); }
	template <> void normL1(const double * pSrc, int len, double * pNorm) { STATUS_CHECK(
		ippsNorm_L1_64f(pSrc, len, pNorm)); }
	template <> void normL1(const int16_t * pSrc, int len, float * pNorm) { STATUS_CHECK(
		ippsNorm_L1_16s32f(pSrc, len, pNorm)); }

	// normL2
	template <> void normL2(const float * pSrc, int len, float * pNorm) { STATUS_CHECK(
		ippsNorm_L2_32f(pSrc, len, pNorm)); }
	template <> void normL2(const double * pSrc, int len, double * pNorm) { STATUS_CHECK(
		ippsNorm_L2_64f(pSrc, len, pNorm)); }
	template <> void normL2(const int16_t * pSrc, int len, float * pNorm) { STATUS_CHECK(
		ippsNorm_L2_16s32f(pSrc, len, pNorm)); }

	// normDiffInf
	template <> void normDiffInf(const float * pSrc1, const float * pSrc2, int len, float * pNorm) { STATUS_CHECK(
		ippsNormDiff_Inf_32f(pSrc1, pSrc2, len, pNorm)); }
	template <> void normDiffInf(const double * pSrc1, const double * pSrc2, int len, double * pNorm) { STATUS_CHECK(
		ippsNormDiff_Inf_64f(pSrc1, pSrc2, len, pNorm)); }
	template <> void normDiffInf(const int16_t * pSrc1, const int16_t * pSrc2, int len, float * pNorm) { STATUS_CHECK(
		ippsNormDiff_Inf_16s32f(pSrc1, pSrc2, len, pNorm)); }

	// normDiffL1
	template <> void normDiffL1(const float * pSrc1, const float * pSrc2, int len, float * pNorm) { STATUS_CHECK(
		ippsNormDiff_L1_32f(pSrc1, pSrc2, len, pNorm)); }
	template <> void normDiffL1(const double * pSrc1, const double * pSrc2, int len, double * pNorm) { STATUS_CHECK(
		ippsNormDiff_L1_64f(pSrc1, pSrc2, len, pNorm)); }
	template <> void normDiffL1(const int16_t * pSrc1, const int16_t * pSrc2, int len, float * pNorm) { STATUS_CHECK(
		ippsNormDiff_L1_16s32f(pSrc1, pSrc2, len, pNorm)); }

	// normDiffL2
	template <> void normDiffL2(const float * pSrc1, const float * pSrc2, int len, float * pNorm) { STATUS_CHECK(
		ippsNormDiff_L2_32f(pSrc1, pSrc2, len, pNorm)); }
	template <> void normDiffL2(const double * pSrc1, const double * pSrc2, int len, double * pNorm) { STATUS_CHECK(
		ippsNormDiff_L2_64f(pSrc1, pSrc2, len, pNorm)); }
	template <> void normDiffL2(const int16_t * pSrc1, const int16_t * pSrc2, int len, float * pNorm) { STATUS_CHECK(
		ippsNormDiff_L2_16s32f(pSrc1, pSrc2, len, pNorm)); }

	// dotProd
	template <> void dotProd(const float * pSrc1, const float * pSrc2, int len, float * pDp) { STATUS_CHECK(
		ippsDotProd_32f(pSrc1, pSrc2, len, pDp)); }
	template <> void dotProd(const float * pSrc1, const float * pSrc2, int len, double * pDp) { STATUS_CHECK(
		ippsDotProd_32f64f(pSrc1, pSrc2, len, pDp)); }
	template <> void dotProd(const double * pSrc1, const double * pSrc2, int len, double * pDp) { STATUS_CHECK(
		ippsDotProd_64f(pSrc1, pSrc2, len, pDp)); }
	template <> void dotProd(const int16_t * pSrc1, const int16_t * pSrc2, int len, int64_t * pDp) { STATUS_CHECK(
		ippsDotProd_16s64s(pSrc1, pSrc2, len, (Ipp64s*)pDp)); }
	template <> void dotProd(const int16_t * pSrc1, const int16_t * pSrc2, int len, float * pDp) { STATUS_CHECK(
		ippsDotProd_16s32f(pSrc1, pSrc2, len, pDp)); }
}

namespace trigonometric
{
	namespace f21
	{
		void cos(const float * pSrc, float * pDst, int len) { STATUS_CHECK(ippsCos_32f_A21(pSrc, pDst, len)); }
		void sin(const float * pSrc, float * pDst, int len) { STATUS_CHECK(ippsSin_32f_A21(pSrc, pDst, len)); }
		void tan(const float * pSrc, float * pDst, int len) { STATUS_CHECK(ippsTan_32f_A21(pSrc, pDst, len)); }
		void acos(const float * pSrc, float * pDst, int len) { STATUS_CHECK(ippsAcos_32f_A21(pSrc, pDst, len)); }
		void asin(const float * pSrc, float * pDst, int len) { STATUS_CHECK(ippsAsin_32f_A21(pSrc, pDst, len)); }
		void atan(const float * pSrc, float * pDst, int len) { STATUS_CHECK(ippsAtan_32f_A21(pSrc, pDst, len)); }
	}

	namespace f24
	{
		void cos(const float * pSrc, float * pDst, int len) { STATUS_CHECK(ippsCos_32f_A24(pSrc, pDst, len)); }
		void sin(const float * pSrc, float * pDst, int len) { STATUS_CHECK(ippsSin_32f_A24(pSrc, pDst, len)); }
		void tan(const float * pSrc, float * pDst, int len) { STATUS_CHECK(ippsTan_32f_A24(pSrc, pDst, len)); }
		void acos(const float * pSrc, float * pDst, int len) { STATUS_CHECK(ippsAcos_32f_A24(pSrc, pDst, len)); }
		void asin(const float * pSrc, float * pDst, int len) { STATUS_CHECK(ippsAsin_32f_A24(pSrc, pDst, len)); }
		void atan(const float * pSrc, float * pDst, int len) { STATUS_CHECK(ippsAtan_32f_A24(pSrc, pDst, len)); }
	}

	namespace d50
	{
		void cos(const double * pSrc, double * pDst, int len) { STATUS_CHECK(ippsCos_64f_A50(pSrc, pDst, len)); }
		void sin(const double * pSrc, double * pDst, int len) { STATUS_CHECK(ippsSin_64f_A50(pSrc, pDst, len)); }
		void tan(const double * pSrc, double * pDst, int len) { STATUS_CHECK(ippsTan_64f_A50(pSrc, pDst, len)); }
		void acos(const double * pSrc, double * pDst, int len) { STATUS_CHECK(ippsAcos_64f_A50(pSrc, pDst, len)); }
		void asin(const double * pSrc, double * pDst, int len) { STATUS_CHECK(ippsAsin_64f_A50(pSrc, pDst, len)); }
		void atan(const double * pSrc, double * pDst, int len) { STATUS_CHECK(ippsAtan_64f_A50(pSrc, pDst, len)); }
	}

	namespace d53
	{
		void cos(const double * pSrc, double * pDst, int len) { STATUS_CHECK(ippsCos_64f_A53(pSrc, pDst, len)); }
		void sin(const double * pSrc, double * pDst, int len) { STATUS_CHECK(ippsSin_64f_A53(pSrc, pDst, len)); }
		void tan(const double * pSrc, double * pDst, int len) { STATUS_CHECK(ippsTan_64f_A53(pSrc, pDst, len)); }
		void acos(const double * pSrc, double * pDst, int len) { STATUS_CHECK(ippsAcos_64f_A53(pSrc, pDst, len)); }
		void asin(const double * pSrc, double * pDst, int len) { STATUS_CHECK(ippsAsin_64f_A53(pSrc, pDst, len)); }
		void atan(const double * pSrc, double * pDst, int len) { STATUS_CHECK(ippsAtan_64f_A53(pSrc, pDst, len)); }
	}
}

} // ipp

