#ifndef _SIMD_H_
#define _SIMD_H_

#include "nosimd.h"
#include "unroll4.h"
#include "sse.h"
#include "sse_ipp.h"

namespace simd
{
#if defined(SIMD_SSE)
	using namespace sse;
#elif defined(SIMD_IPP)
# ifdef IPP_FIXED_ACCURACY
	using namespace ipp_f24_d53;
# else
	using namespace ipp;
	using namespace ipp_f24_d53::trigonometric;
# endif
#elif defined(SIMD_UNROLL)
	using namespace unroll;
#else
	using namespace nosimd;
#endif
}

#endif

