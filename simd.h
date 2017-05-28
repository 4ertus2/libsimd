#pragma once
#include "nosimd.h"

#if defined(NO_SIMD)
namespace simd { using namespace nosimd; }
#elif (defined(__amd64__) || defined(__i386__) || defined(_M_AMD64))

#if defined(SIMD_IPP)
#include "sse_ipp.h"
namespace simd { using namespace ipp; }
#elif defined(SIMD_AVX)
#include "avx-float.h"
#include "avx-double.h"
#include "avx-int.h"
namespace simd { using namespace sse; }
#else
#include "sse-float.h"
#include "sse-double.h"
#include "sse-i128.h"
namespace simd { using namespace sse; }
#endif

#elif defined(__arm__)
#include "neon.h"
namespace simd { using namespace neon; }
#else
namespace simd { using namespace nosimd; }
#endif
