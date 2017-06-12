#pragma once
#include <cstdint>
#include <cmath>

namespace
{
    template<typename T> T sqrt_cmath(T x) { return sqrt(float(x)); }
    template<> double sqrt_cmath(double x) { return sqrt(x); }

    template<typename T> T exp_cmath(T x) { return exp(float(x)); }
    template<> double exp_cmath(double x) { return exp(x); }

    template<typename T> T log_cmath(T x) { return log(float(x)); }
    template<> double log_cmath(double x) { return log(x); }

    template<typename T> T pow_cmath(T x, T y) { return pow(float(x), float(y)); }
    template<> double pow_cmath(double x, double y) { return pow(x, y); }

    template<typename T> T cbrt_cmath(T x) { return cbrt(float(x)); }
    template<> double cbrt_cmath(double x) { return cbrt(x); }

    template<typename T> T hypot_cmath(T x, T y) { return hypot(float(x), float(y)); }
    template<> double hypot_cmath(double x, double y) { return hypot(x, y); }

    //

    template<typename T> T sin_cmath(T x) { return sin(float(x)); }
    template<> double sin_cmath(double x) { return sin(x); }

    template<typename T> T cos_cmath(T x) { return cos(float(x)); }
    template<> double cos_cmath(double x) { return cos(x); }

    template<typename T> T tan_cmath(T x) { return tan(float(x)); }
    template<> double tan_cmath(double x) { return tan(x); }

    template<typename T> T asin_cmath(T x) { return asin(float(x)); }
    template<> double asin_cmath(double x) { return asin(x); }

    template<typename T> T acos_cmath(T x) { return acos(float(x)); }
    template<> double acos_cmath(double x) { return acos(x); }

    template<typename T> T atan_cmath(T x) { return atan(float(x)); }
    template<> double atan_cmath(double x) { return atan(x); }
}

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

    namespace compare
    {
        template<typename _T> void find(_T * pSrc, _T val, int len, int * pPosition)
        {
            int& i = *pPosition;
            for (i = 0; i < len; ++i)
            {
                if (pSrc[i] == val)
                    break;
            }
        }

        template<typename _T> void findDiff(const _T * pSrc1, _T * pSrc2, int len, int * pPosition)
        {
            int& i = *pPosition;
            for (i = 0; i < len; ++i)
            {
                if (pSrc1[i] != pSrc2[i])
                    break;
            }
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

        template<typename T> void sqrt(const T * pSrc, T * pDst, int len)
        {
            for (int i=0; i < len; ++i)
                pDst[i] = sqrt_cmath(pSrc[i]);
        }

        template<typename T> void invSqrt(const T * pSrc, T * pDst, int len)
        {
            for (int i=0; i < len; ++i)
                pDst[i] = 1.0f/sqrt_cmath(pSrc[i]);
        }

        template<typename T> void powx(const T* pSrc, const T constValue, T* pDst, int len)
        {
            for (int i=0; i < len; ++i)
                pDst[i] = pow_cmath(pSrc[i], constValue);
        }

        template<typename T> void pow(const T* pSrc1, const T* pSrc2, T* pDst, int len)
        {
            for (int i=0; i < len; ++i)
                pDst[i] = pow_cmath(pSrc1[i], pSrc2[i]);
        }

        template<typename T> void cbrt(const T * pSrc, T * pDst, int len)
        {
            for (int i=0; i < len; ++i)
                pDst[i] = cbrt_cmath(pSrc[i]);
        }

        template<typename T> void hypot(const T* pSrc1, const T* pSrc2, T* pDst, int len)
        {
            for (int i=0; i < len; ++i)
                pDst[i] = hypot_cmath(pSrc1[i], pSrc2[i]);
        }
    }

    namespace exp_log
    {
        template<typename T> void exp(const T * pSrc, T * pDst, int len)
        {
            for (int i=0; i < len; ++i)
                pDst[i] = exp_cmath(pSrc[i]);
        }

        template<typename T> void ln(const T * pSrc, T * pDst, int len)
        {
            for (int i=0; i < len; ++i)
                pDst[i] = log_cmath(pSrc[i]);
        }
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

        template<typename T, typename U> void meanStdDev(const T * pSrc, int len, U * pMean, U * pStdDev)
        {
            mean(pSrc, len, pMean);

            U s = 0;
            for (int i=0; i < len; ++i)
            {
                U x = pSrc[i] - *pMean;
                s += x * x;
            }
            *pStdDev = sqrt_cmath(s/(len-1));
        }

        template<typename T, typename U> void stdDev(const T * pSrc, int len, U * pStdDev)
        {
            U m;
            meanStdDev(pSrc, len, &m, pStdDev);
        }

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

        template<typename T, typename U> void normL2(const T * pSrc, int len, U * pNorm)
        {
            U norm = 0;
            for (int i=0; i < len; ++i)
            {
                norm += pSrc[i] * pSrc[i];
            }
            *pNorm = sqrt_cmath(norm);
        }

        template<typename T, typename U> void normDiffInf(const T * pSrc1, const T * pSrc2, int len, U * pNorm)
        {
            T x = pSrc1[0]-pSrc2[0];
            U mx = x;
            if (x < 0)
                mx = -x;

            for (int i=1; i < len; ++i)
            {
                x = pSrc1[i] - pSrc2[i];
                if (x < 0)
                {
                    if (mx < -x)
                        mx = -x;
                }
                else
                {
                    if (mx < x)
                        mx = x;
                }
            }
            *pNorm = mx;
        }

        template<typename T, typename U> void normDiffL1(const T * pSrc1, const T * pSrc2, int len, U * pNorm)
        {
            U norm = 0;
            for (int i=0; i < len; ++i)
            {
                T x = pSrc1[i] - pSrc2[i];
                if (x < 0)
                    norm -= x;
                else
                    norm += x;
            }
            *pNorm = norm;
        }

        template<typename T, typename U> void normDiffL2(const T * pSrc1, const T * pSrc2, int len, U * pNorm)
        {
            U norm = 0;
            for (int i=0; i < len; ++i)
            {
                U x = pSrc1[i] - pSrc2[i];
                norm += x * x;
            }
            *pNorm = sqrt_cmath(norm);
        }

        template<typename T, typename U> void dotProd(const T * pSrc1, const T * pSrc2, int len, U * pDp)
        {
            U s = 0;
            for (int i=0; i < len; ++i)
                s += pSrc1[i] * pSrc2[i];
            *pDp = s;
        }
    }

    namespace trigonometric
    {
        template<typename T> void sin(const T * pSrc, T * pDst, int len)
        {
            for (int i=0; i < len; ++i)
                pDst[i] = sin_cmath(pSrc[i]);
        }

        template<typename T> void cos(const T * pSrc, T * pDst, int len)
        {
            for (int i=0; i < len; ++i)
                pDst[i] = cos_cmath(pSrc[i]);
        }

        template<typename T> void tan(const T * pSrc, T * pDst, int len)
        {
            for (int i=0; i < len; ++i)
                pDst[i] = tan_cmath(pSrc[i]);
        }

        template<typename T> void asin(const T * pSrc, T * pDst, int len)
        {
            for (int i=0; i < len; ++i)
                pDst[i] = asin_cmath(pSrc[i]);
        }

        template<typename T> void acos(const T * pSrc, T * pDst, int len)
        {
            for (int i=0; i < len; ++i)
                pDst[i] = acos_cmath(pSrc[i]);
        }

        template<typename T> void atan(const T * pSrc, T * pDst, int len)
        {
            for (int i=0; i < len; ++i)
                pDst[i] = atan_cmath(pSrc[i]);
        }
    }

    using namespace nosimd::common;
    using namespace nosimd::compare;
    using namespace nosimd::arithmetic;
    using namespace nosimd::power;
    using namespace nosimd::exp_log;
    using namespace nosimd::statistical;
    using namespace nosimd::trigonometric;
}
