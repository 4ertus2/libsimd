// $Id$

#ifndef _SSE_IPP_FFT_H_
#define _SSE_IPP_FFT_H_

#include <ipps.h>

#include "ipp_exception.h"

#define IPPS_FFT_HINT ippAlgHintFast

namespace ipps
{
	///
	class FFT
	{
	public:
		typedef enum {
			NODIV_BY_ANY = IPP_FFT_NODIV_BY_ANY,	///< 1		1
			DIV_FWD_BY_N = IPP_FFT_DIV_FWD_BY_N,	///< 1/N	1
			DIV_INV_BY_N = IPP_FFT_DIV_INV_BY_N,	///< 1		1/N
			DIV_BY_SQRTN = IPP_FFT_DIV_BY_SQRTN 	///< 1/sqrt(N)
		} DivCoef;

		FFT(int fftOrder, DivCoef divFlag = DIV_FWD_BY_N)
		:	pFFT_(0)
		{
			STATUS_CHECK( ippsFFTInitAlloc_R_32f(&pFFT_, fftOrder, divFlag, IPPS_FFT_HINT) );
		}

		~FFT()
		{
			if (pFFT_)
				ippsFFTFree_R_32f(pFFT_);
		}

		void convert(const float* pSrc, float* pDst)
		{
			STATUS_CHECK( ippsFFTFwd_RToCCS_32f(pSrc, pDst, pFFT_, 0) );
		}

	private:
		IppsFFTSpec_R_32f * pFFT_;
	};

	///
	class FwdDCT
	{
	public:
		FwdDCT(int length)
		:	pDCT_(0)
		{
			STATUS_CHECK( ippsDCTFwdInitAlloc_32f(&pDCT_, length, IPPS_FFT_HINT) );
		}

		~FwdDCT()
		{
			if (pDCT_)
				ippsDCTFwdFree_32f(pDCT_);
		}

		void convert(const float* pSrc, float* pDst)
		{
			STATUS_CHECK( ippsDCTFwd_32f(pSrc, pDst, pDCT_, 0) );
		}

	private:
		IppsDCTFwdSpec_32f * pDCT_;
	};

	///
	class InvDCT
	{
	public:
		InvDCT(int length)
		:	pDCT_(0)
		{
			STATUS_CHECK( ippsDCTInvInitAlloc_32f(&pDCT_, length, IPPS_FFT_HINT) );
		}

		~InvDCT()
		{
			if (pDCT_)
				ippsDCTInvFree_32f(pDCT_);
		}

		void convert(const float* pSrc, float* pDst)
		{
			STATUS_CHECK( ippsDCTInv_32f(pSrc, pDst, pDCT_, 0) );
		}

	private:
		IppsDCTInvSpec_32f * pDCT_;
	};
}

#endif
