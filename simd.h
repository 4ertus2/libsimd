#pragma once
#include "nosimd.h"

// NEON
#if defined(__arm__) && !defined(NO_NEON)
#include "neon.h"
namespace simd { using namespace neon; }
// IPP
#elif defined(SIMD_IPP)
#include "sse_ipp.h"
namespace simd { using namespace ipp; }
// SSE
#elif (defined(__amd64__) || defined(__i386__) || defined(_M_AMD64)) && !defined(NO_SSE)
#include "sse-float.h"
#include "sse-double.h"
#include "sse-i128.h"
namespace simd { using namespace sse; }
// NO SIMD
#else
namespace simd { using namespace nosimd; }
#endif
