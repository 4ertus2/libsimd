// $Id$

#ifndef _SSE_VARIANT_
#define _SSE_VARIANT_

#include "sse_ipp.h"

namespace simd
{
#ifdef IPP_FIXED_ACCURACY
	using namespace ipp_f24_d53;
#else
	using namespace ipp;
	using namespace ipp_f24_d53::trigonometric;
#endif
}

#endif
