#ifndef _SIMD_NOSIMD_H_
#define _SIMD_NOSIMD_H_

namespace nosimd
{
	namespace common
	{
		template<typename T> T * malloc(int len) { return new T[len]; }
		template<typename T> void free(T * ptr) { delete ptr; }

		template<typename T> void zero(T * pDst, int len)
		{
			for (int i=0; i<len; ++i)
				pDst[i] = 0;
		}
	
		template<typename T> void set(T val, T * pDst, int len)
		{
			for (int i=0; i<len; ++i)
				pDst[i] = val;
		}
	
		template<typename T> void copy(const T * pSrc, T * pDst, int len)
		{
			for (int i=0; i<len; ++i)
				pDst[i] = pSrc[i];
		}

		template<typename T> void move(const T * pSrc, T * pDst, int len)
		{
			for (int i=0; i < len; ++i)
				pDst[i] = pSrc[i];
		}
		
		template<typename T, typename U> void convert(const T * pSrc, U * pDst, int len)
		{
			for (int i=0; i < len; ++i)
				pDst[i] = pSrc[i];
		}
	}

	namespace arithmetic
	{
		template<typename T> void addC(const T * pSrc, T val, T * pDst, int len)
		{
			if (pSrc == pDst)
			{
				for (int i=0; i < len; ++i)
					pDst[i] += val;
			}
			else
			{
				for (int i=0; i < len; ++i)
					pDst[i] = pSrc[i] + val;
			}
		}
		
		template<typename T> void add(const T * pSrc1, const T * pSrc2, T * pDst, int len)
		{
			if (pSrc1 == pDst)
			{
				for (int i=0; i < len; ++i)
					pDst[i] += pSrc2[i];
			}
			else if(pSrc2 == pDst)
			{
				for (int i=0; i < len; ++i)
					pDst[i] += pSrc1[i];
			}
			else
			{ 
				for (int i=0; i < len; ++i)
					pDst[i] = pSrc1[i] + pSrc2[i];
			}
		}
		
		template<typename T> void subC(const T * pSrc, T val, T * pDst, int len)
		{
			if (pSrc == pDst)
			{
				for (int i=0; i < len; ++i)
					pDst[i] -= val;
			}
			else
			{
				for (int i=0; i < len; ++i)
					pDst[i] = pSrc[i] - val;
			}
		}
		
		template<typename T> void subCRev(const T * pSrc, T val, T * pDst, int len)
		{
			for (int i=0; i < len; ++i)
				pDst[i] = val - pSrc[i];
		}
		
		template<typename T> void sub(const T * pSrc1, const T * pSrc2, T * pDst, int len)
		{
			if (pSrc1 == pDst)
			{
				for (int i=0; i < len; ++i)
					pDst[i] -= pSrc2[i];
			}
			else
			{
				for (int i=0; i < len; ++i)
					pDst[i] = pSrc1[i] - pSrc2[i];
			}
		}

		template<typename T> void mulC(const T * pSrc, T val, T * pDst, int len)
		{
			if (pSrc == pDst)
			{
				for (int i=0; i < len; ++i)
					pDst[i] *= val;
			}
			else
			{
				for (int i=0; i < len; ++i)
					pDst[i] = pSrc[i] * val;
			}
		}

		template<typename T> void mul(const T * pSrc1, const T * pSrc2, T * pDst, int len)
		{
			if (pSrc1 == pDst)
			{
				for (int i=0; i < len; ++i)
					pDst[i] *= pSrc2[i];
			}
			else if(pSrc2 == pDst)
			{
				for (int i=0; i < len; ++i)
					pDst[i] *= pSrc1[i];
			}
			else
			{ 
				for (int i=0; i < len; ++i)
					pDst[i] = pSrc1[i] * pSrc2[i];
			}
		}

		template<typename T> void divC(const T * pSrc, T val, T * pDst, int len)
		{
			if (pSrc == pDst)
			{
				for (int i=0; i < len; ++i)
					pDst[i] /= val;
			}
			else
			{
				for (int i=0; i < len; ++i)
					pDst[i] = pSrc[i] / val;
			}
		}

		template<typename T> void divCRev(const T * pSrc, T val, T * pDst, int len)
		{
			for (int i=0; i < len; ++i)
				pDst[i] = val / pSrc[i];
		}

		template<typename T> void div(const T * pSrc1, const T * pSrc2, T * pDst, int len)
		{
			if (pSrc1 == pDst)
			{
				for (int i=0; i < len; ++i)
					pDst[i] /= pSrc2[i];
			}
			else
			{
				for (int i=0; i < len; ++i)
					pDst[i] = pSrc1[i] / pSrc2[i];
			}
		}

		template<typename T> void abs(const T * pSrc, T * pDst, int len)
		{
			if (pSrc == pDst)
			{
				for (int i=0; i < len; ++i)
					if (pDst[i] < 0)
						pDst[i] = -pDst[i];
			}
			else
			{
				for (int i=0; i < len; ++i)
				{
					if (pSrc[i] >= 0)
						pDst[i] = pSrc[i];
					else
						pDst[i] = -pSrc[i];
				}
			}
		}
	}
	
	namespace power
	{
		template<typename T> void inv(const T * pSrc, T * pDst, int len)
		{
			for (int i=0; i < len; ++i)
				pDst[i] = T(1) / pSrc[i];
		}
		
		template<typename T> void sqr(const T * pSrc, T * pDst, int len)
		{
			for (int i=0; i < len; ++i)
				pDst[i] = pSrc[i] * pSrc[i];
		}
		
		// sqrt
		// powx
		// pow
	}
	
	namespace exp_log
	{
		// exp
		// log
	}
	
	namespace statistical
	{
		template<typename T> void max(const T * pSrc, int len, T * pMax)
		{
			T mx = pSrc[0];
			for (int i=1; i < len; ++i)
				if (pSrc[i] > mx)
					mx = pSrc[i];
			*pMax = mx;
		}

		template<typename T> void maxIndx(const T * pSrc, int len, T * pMax, int * pIndx)
		{
			T mx = pSrc[0];
			int idx = 0;
			for (int i=1; i < len; ++i)
				if (pSrc[i] > mx) {
					mx = pSrc[i];
					idx = i;
				}
			*pMax = mx;
			*pIndx = idx;
		}

		template<typename T> void min(const T * pSrc, int len, T * pMin)
		{
			T mn = pSrc[0];
			for (int i=1; i < len; ++i)
				if (pSrc[i] < mn)
					mn = pSrc[i];
			*pMin = mn;
		}

		template<typename T> void minIndx(const T * pSrc, int len, T * pMin, int * pIndx)
		{
			T mn = pSrc[0];
			int idx = 0;
			for (int i=1; i < len; ++i)
				if (pSrc[i] < mn) {
					mn = pSrc[i];
					idx = i;
				}
			*pMin = mn;
			*pIndx = idx;
		}

		template<typename T> void minMax(const T * pSrc, int len, T * pMin, T * pMax)
		{
			T mn = pSrc[0];
			T mx = pSrc[0];
			for (int i=1; i < len; ++i)
			{
				if (pSrc[i] < mn)
					mn = pSrc[i];
				if (pSrc[i] > mx)
					mx = pSrc[i];
			}
			*pMin = mn;
			*pMax = mx;
		}

		template<typename T> void minMaxIndx(const T * pSrc, int len, T * pMin, int * pMinIndx, T * pMax, int * pMaxIndx)
		{
			T mn = pSrc[0];
			T mx = pSrc[0];
			int idxMin = 0;
			int idxMax = 0;
			for (int i=1; i < len; ++i)
			{
				if (pSrc[i] < mn) {
					mn = pSrc[i];
					idxMin = i;
				}
				if (pSrc[i] > mx) {
					mx = pSrc[i];
					idxMax = i;
				}
			}
			*pMin = mn;
			*pMinIndx = idxMin;
			*pMax = mx;
			*pMaxIndx = idxMax;
		}
		
		template<typename T, typename U> void sum(const T * pSrc, int len, U * pSum)
		{
			U s = 0;
			for (int i=0; i < len; ++i)
				s += pSrc[i];
			*pSum = s;
		}

		template<typename T, typename U> void mean(const T * pSrc, int len, U * pMean)
		{
			U s = 0;
			for (int i=0; i < len; ++i)
				s += pSrc[i];
			*pMean = s/len;
		}
		
		// stdDev
		// meanStdDev
		
		template<typename T, typename U> void normInf(const T * pSrc, int len, U * pNorm)
		{
			U mx = pSrc[0];
			if (pSrc[0] < 0)
				mx = -pSrc[0];

			for (int i=1; i < len; ++i)
			{
				if (pSrc[i] < 0)
				{
					if (mx < -pSrc[i])
						mx = -pSrc[i];
				}
				else
				{
					if (mx < pSrc[i])
						mx = pSrc[i];
				}
			}
			*pNorm = mx;
		}

		template<typename T, typename U> void normL1(const T * pSrc, int len, U * pNorm)
		{
			U norm = 0;
			for (int i=0; i < len; ++i)
			{
				if (pSrc[i] < 0)
					norm -= pSrc[i];
				else
					norm += pSrc[i];
			}
			*pNorm = norm;
		}

		// normL2

		template<typename T, typename U> void dotProd(const T * pSrc1, const T * pSrc2, int len, U * pDp)
		{
			U s = 0;
			for (int i=0; i < len; ++i)
				s += pSrc1[i] * pSrc2[i];
			*pDp = s;
		}
	}

	using namespace nosimd::common;
	using namespace nosimd::arithmetic;
	using namespace nosimd::power;
	using namespace nosimd::exp_log;
	using namespace nosimd::statistical;
}

#endif

