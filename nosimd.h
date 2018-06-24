#pragma once
#include <cstdint>
#include <cmath>
#include <exception>
#include <string>

namespace simd
{
    ///
    class Exception : public std::exception
    {
    public:
        Exception(const char * file, uint32_t line, const char * func, int32_t err = 0)
        :   file_(file), line_(line), function_(func), errorCode_(err)
        {}

        Exception(const char * file, uint32_t line, const char * func, int32_t err, const char * msg)
        :   file_(file), line_(line), function_(func), errorCode_(err), what_(msg)
        {}

        const char * what() const noexcept { return what_.c_str(); }
        const char * file() const noexcept { return file_; }
        uint32_t line() const noexcept { return line_; }
        int32_t code() const noexcept { return errorCode_; }

    private:
        const char * file_;
        uint32_t line_;
        const char * function_;
        int32_t errorCode_;
        std::string what_;
    };
}

namespace
{
    template<typename _T> inline _T sqrt_cmath(_T x) { return sqrt(float(x)); }
    template<> inline double sqrt_cmath(double x) { return sqrt(x); }

    template<typename _T> inline _T exp_cmath(_T x) { return exp(float(x)); }
    template<> inline double exp_cmath(double x) { return exp(x); }

    template<typename _T> inline _T log_cmath(_T x) { return log(float(x)); }
    template<> inline double log_cmath(double x) { return log(x); }

    template<typename _T> inline _T pow_cmath(_T x, _T y) { return pow(float(x), float(y)); }
    template<> inline double pow_cmath(double x, double y) { return pow(x, y); }

    template<typename _T> inline _T cbrt_cmath(_T x) { return cbrt(float(x)); }
    template<> inline double cbrt_cmath(double x) { return cbrt(x); }

    template<typename _T> inline _T hypot_cmath(_T x, _T y) { return hypot(float(x), float(y)); }
    template<> inline double hypot_cmath(double x, double y) { return hypot(x, y); }

    //

    template<typename _T> inline _T sin_cmath(_T x) { return sin(float(x)); }
    template<> inline double sin_cmath(double x) { return sin(x); }

    template<typename _T> inline _T cos_cmath(_T x) { return cos(float(x)); }
    template<> inline double cos_cmath(double x) { return cos(x); }

    template<typename _T> inline _T tan_cmath(_T x) { return tan(float(x)); }
    template<> inline double tan_cmath(double x) { return tan(x); }

    template<typename _T> inline _T asin_cmath(_T x) { return asin(float(x)); }
    template<> inline double asin_cmath(double x) { return asin(x); }

    template<typename _T> inline _T acos_cmath(_T x) { return acos(float(x)); }
    template<> inline double acos_cmath(double x) { return acos(x); }

    template<typename _T> inline _T atan_cmath(_T x) { return atan(float(x)); }
    template<> inline double atan_cmath(double x) { return atan(x); }
}

namespace nosimd
{
    namespace common
    {
        template<typename _T> inline _T * malloc(int len) { return new _T[len]; }
        template<typename _T> inline void free(_T * ptr) { delete ptr; }

        template<typename _T>
        inline void zero(_T * pDst, int len)
        {
            for (int i = 0; i < len; ++i)
                pDst[i] = 0;
        }

        template<typename _T>
        inline void set(_T val, _T * pDst, int len)
        {
            for (int i = 0; i < len; ++i)
                pDst[i] = val;
        }

        template<typename _T>
        inline void copy(const _T * pSrc, _T * pDst, int len)
        {
            for (int i = 0; i < len; ++i)
                pDst[i] = pSrc[i];
        }

        template<typename _T>
        inline void move(const _T * pSrc, _T * pDst, int len)
        {
            for (int i = 0; i < len; ++i)
                pDst[i] = pSrc[i];
        }

        template<typename _T, typename _U>
        inline void convert(const _T * pSrc, _U * pDst, int len)
        {
            for (int i = 0; i < len; ++i)
                pDst[i] = pSrc[i];
        }
    }

    namespace compare
    {
        template<typename _T>
        inline void find(_T * pSrc, _T val, int len, int * pPosition)
        {
            int& i = *pPosition;
            for (i = 0; i < len; ++i)
            {
                if (pSrc[i] == val)
                    break;
            }
        }

        template<typename _T>
        inline void findNot(_T * pSrc, _T val, int len, int * pPosition)
        {
            int& i = *pPosition;
            for (i = 0; i < len; ++i)
            {
                if (pSrc[i] != val)
                    break;
            }
        }

        template<typename _T>
        inline void findSame(const _T * pSrc1, _T * pSrc2, int len, int * pPosition)
        {
            int& i = *pPosition;
            for (i = 0; i < len; ++i)
            {
                if (pSrc1[i] == pSrc2[i])
                    break;
            }
        }

        template<typename _T>
        inline void findDiff(const _T * pSrc1, _T * pSrc2, int len, int * pPosition)
        {
            int& i = *pPosition;
            for (i = 0; i < len; ++i)
            {
                if (pSrc1[i] != pSrc2[i])
                    break;
            }
        }

        // TODO: rfind*
    }

    namespace arithmetic
    {
        template<typename _T>
        inline void addC(const _T * pSrc, _T val, _T * pDst, int len)
        {
            if (pSrc == pDst)
            {
                for (int i = 0; i < len; ++i)
                    pDst[i] += val;
            }
            else
            {
                for (int i = 0; i < len; ++i)
                    pDst[i] = pSrc[i] + val;
            }
        }

        template<typename _T>
        inline void add(const _T * pSrc1, const _T * pSrc2, _T * pDst, int len)
        {
            if (pSrc1 == pDst)
            {
                for (int i = 0; i < len; ++i)
                    pDst[i] += pSrc2[i];
            }
            else if(pSrc2 == pDst)
            {
                for (int i = 0; i < len; ++i)
                    pDst[i] += pSrc1[i];
            }
            else
            {
                for (int i = 0; i < len; ++i)
                    pDst[i] = pSrc1[i] + pSrc2[i];
            }
        }

        template<typename _T>
        inline void subC(const _T * pSrc, _T val, _T * pDst, int len)
        {
            if (pSrc == pDst)
            {
                for (int i = 0; i < len; ++i)
                    pDst[i] -= val;
            }
            else
            {
                for (int i = 0; i < len; ++i)
                    pDst[i] = pSrc[i] - val;
            }
        }

        template<typename _T>
        inline void subCRev(const _T * pSrc, _T val, _T * pDst, int len)
        {
            for (int i = 0; i < len; ++i)
                pDst[i] = val - pSrc[i];
        }

        template<typename _T>
        inline void sub(const _T * pSrc1, const _T * pSrc2, _T * pDst, int len)
        {
            if (pSrc1 == pDst)
            {
                for (int i = 0; i < len; ++i)
                    pDst[i] -= pSrc2[i];
            }
            else
            {
                for (int i = 0; i < len; ++i)
                    pDst[i] = pSrc1[i] - pSrc2[i];
            }
        }

        template<typename _T>
        inline void mulC(const _T * pSrc, _T val, _T * pDst, int len)
        {
            if (pSrc == pDst)
            {
                for (int i = 0; i < len; ++i)
                    pDst[i] *= val;
            }
            else
            {
                for (int i = 0; i < len; ++i)
                    pDst[i] = pSrc[i] * val;
            }
        }

        template<typename _T>
        inline void mul(const _T * pSrc1, const _T * pSrc2, _T * pDst, int len)
        {
            if (pSrc1 == pDst)
            {
                for (int i = 0; i < len; ++i)
                    pDst[i] *= pSrc2[i];
            }
            else if(pSrc2 == pDst)
            {
                for (int i = 0; i < len; ++i)
                    pDst[i] *= pSrc1[i];
            }
            else
            {
                for (int i = 0; i < len; ++i)
                    pDst[i] = pSrc1[i] * pSrc2[i];
            }
        }

        template<typename _T>
        inline void divC(const _T * pSrc, _T val, _T * pDst, int len)
        {
            if (pSrc == pDst)
            {
                for (int i = 0; i < len; ++i)
                    pDst[i] /= val;
            }
            else
            {
                for (int i = 0; i < len; ++i)
                    pDst[i] = pSrc[i] / val;
            }
        }

        template<typename _T>
        inline void divCRev(const _T * pSrc, _T val, _T * pDst, int len)
        {
            for (int i = 0; i < len; ++i)
                pDst[i] = val / pSrc[i];
        }

        template<typename _T>
        inline void div(const _T * pSrc1, const _T * pSrc2, _T * pDst, int len)
        {
            if (pSrc1 == pDst)
            {
                for (int i = 0; i < len; ++i)
                    pDst[i] /= pSrc2[i];
            }
            else
            {
                for (int i = 0; i < len; ++i)
                    pDst[i] = pSrc1[i] / pSrc2[i];
            }
        }

        template<typename _T>
        inline void abs(const _T * pSrc, _T * pDst, int len)
        {
            if (pSrc == pDst)
            {
                for (int i = 0; i < len; ++i)
                    if (pDst[i] < 0)
                        pDst[i] = -pDst[i];
            }
            else
            {
                for (int i = 0; i < len; ++i)
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
        template<typename _T>
        inline void inv(const _T * pSrc, _T * pDst, int len)
        {
            for (int i = 0; i < len; ++i)
                pDst[i] = _T(1) / pSrc[i];
        }

        template<typename _T>
        inline void sqrt(const _T * pSrc, _T * pDst, int len)
        {
            for (int i = 0; i < len; ++i)
                pDst[i] = sqrt_cmath(pSrc[i]);
        }

        template<typename _T>
        inline void invSqrt(const _T * pSrc, _T * pDst, int len)
        {
            for (int i = 0; i < len; ++i)
                pDst[i] = 1.0f/sqrt_cmath(pSrc[i]);
        }

        template<typename _T>
        inline void powx(const _T * pSrc, const _T constValue, _T * pDst, int len)
        {
            for (int i = 0; i < len; ++i)
                pDst[i] = pow_cmath(pSrc[i], constValue);
        }

        template<typename _T>
        inline void pow(const _T * pSrc1, const _T * pSrc2, _T * pDst, int len)
        {
            for (int i = 0; i < len; ++i)
                pDst[i] = pow_cmath(pSrc1[i], pSrc2[i]);
        }

        template<typename _T>
        inline void cbrt(const _T * pSrc, _T * pDst, int len)
        {
            for (int i = 0; i < len; ++i)
                pDst[i] = cbrt_cmath(pSrc[i]);
        }

        template<typename _T>
        inline void hypot(const _T * pSrc1, const _T * pSrc2, _T * pDst, int len)
        {
            for (int i = 0; i < len; ++i)
                pDst[i] = hypot_cmath(pSrc1[i], pSrc2[i]);
        }
    }

    namespace exp_log
    {
        template<typename _T>
        inline void exp(const _T * pSrc, _T * pDst, int len)
        {
            for (int i = 0; i < len; ++i)
                pDst[i] = exp_cmath(pSrc[i]);
        }

        template<typename _T>
        inline void ln(const _T * pSrc, _T * pDst, int len)
        {
            for (int i = 0; i < len; ++i)
                pDst[i] = log_cmath(pSrc[i]);
        }
    }

    namespace statistical
    {
        template<typename _T>
        inline void maxIndx(const _T * pSrc, int len, _T * pMax, int * pIndx)
        {
            _T& maxVal = *pMax;
            maxVal = pSrc[0];
            for (int i=1; i < len; ++i)
            {
                if (pSrc[i] > maxVal) {
                    maxVal = pSrc[i];
                    *pIndx = i;
                }
            }
        }

        template<typename _T>
        inline void minIndx(const _T * pSrc, int len, _T * pMin, int * pIndx)
        {
            _T& minVal = *pMin;
            minVal = pSrc[0];
            for (int i=1; i < len; ++i)
            {
                if (pSrc[i] < minVal) {
                    minVal = pSrc[i];
                    *pIndx = i;
                }
            }
        }

        template<typename _T>
        inline void minMaxIndx(const _T * pSrc, int len, _T * pMin, int * pMinIndx, _T * pMax, int * pMaxIndx)
        {
            _T& minVal = *pMin;
            _T& maxVal = *pMax;
            minVal = maxVal = pSrc[0];
            for (int i=1; i < len; ++i)
            {
                if (pSrc[i] < minVal) {
                    minVal = pSrc[i];
                    *pMinIndx = i;
                }
                if (pSrc[i] > maxVal) {
                    maxVal = pSrc[i];
                    *pMaxIndx = i;
                }
            }
        }

        template<typename _T>
        inline void max(const _T * pSrc, int len, _T * pMax)
        {
            int pos;
            maxIndx(pSrc, len, pMax, &pos);
        }

        template<typename _T>
        inline void min(const _T * pSrc, int len, _T * pMin)
        {
            int pos;
            minIndx(pSrc, len, pMin, &pos);
        }

        template<typename _T>
        inline void minMax(const _T * pSrc, int len, _T * pMin, _T * pMax)
        {
            int posMin, posMax;
            minMaxIndx(pSrc, len, pMin, &posMin, pMax, &posMax);
        }

        template<typename _T, typename _U>
        inline void sum(const _T * pSrc, int len, _U * pSum)
        {
            *pSum = 0;
            for (int i = 0; i < len; ++i)
                *pSum += pSrc[i];
        }

        template<typename _T, typename _U>
        inline void mean(const _T * pSrc, int len, _U * pMean)
        {
            *pMean = 0;
            for (int i = 0; i < len; ++i)
                *pMean += pSrc[i];
            *pMean /= len;
        }

        template<typename _T, typename _U>
        inline void meanStdDev(const _T * pSrc, int len, _U * pMean, _U * pStdDev)
        {
            mean(pSrc, len, pMean);

            _U s = 0;
            for (int i = 0; i < len; ++i)
            {
                _U x = pSrc[i] - *pMean;
                s += x * x;
            }
            *pStdDev = sqrt_cmath(s/(len-1));
        }

        template<typename _T, typename _U>
        inline void stdDev(const _T * pSrc, int len, _U * pStdDev)
        {
            _U m;
            meanStdDev(pSrc, len, &m, pStdDev);
        }

        template<typename _T, typename _U>
        inline void dotProd(const _T * pSrc1, const _T * pSrc2, int len, _U * pDp)
        {
            *pDp = 0;
            for (int i = 0; i < len; ++i)
                *pDp += pSrc1[i] * pSrc2[i];
        }

        template<typename _T, typename _U>
        inline void normL2(const _T * pSrc, int len, _U * pNorm)
        {
            dotProd(pSrc, pSrc, len, pNorm);
            *pNorm = sqrt_cmath(*pNorm);
        }

        template<typename _T, typename _U>
        inline void normInf(const _T * pSrc, int len, _U * pNorm)
        {
            _U mx = pSrc[0];
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

        template<typename _T, typename _U>
        inline void normL1(const _T * pSrc, int len, _U * pNorm)
        {
            _U norm = 0;
            for (int i = 0; i < len; ++i)
            {
                if (pSrc[i] < 0)
                    norm -= pSrc[i];
                else
                    norm += pSrc[i];
            }
            *pNorm = norm;
        }

        template<typename _T, typename _U>
        inline void normDiffInf(const _T * pSrc1, const _T * pSrc2, int len, _U * pNorm)
        {
            _T x = pSrc1[0]-pSrc2[0];
            _U mx = x;
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

        template<typename _T, typename _U>
        inline void normDiffL1(const _T * pSrc1, const _T * pSrc2, int len, _U * pNorm)
        {
            _U norm = 0;
            for (int i = 0; i < len; ++i)
            {
                _T x = pSrc1[i] - pSrc2[i];
                if (x < 0)
                    norm -= x;
                else
                    norm += x;
            }
            *pNorm = norm;
        }

        template<typename _T, typename _U>
        inline void normDiffL2(const _T * pSrc1, const _T * pSrc2, int len, _U * pNorm)
        {
            _U norm = 0;
            for (int i = 0; i < len; ++i)
            {
                _U x = pSrc1[i] - pSrc2[i];
                norm += x * x;
            }
            *pNorm = sqrt_cmath(norm);
        }
    }

    namespace trigonometric
    {
        template<typename _T>
        inline void sin(const _T * pSrc, _T * pDst, int len)
        {
            for (int i = 0; i < len; ++i)
                pDst[i] = sin_cmath(pSrc[i]);
        }

        template<typename _T>
        inline void cos(const _T * pSrc, _T * pDst, int len)
        {
            for (int i = 0; i < len; ++i)
                pDst[i] = cos_cmath(pSrc[i]);
        }

        template<typename _T>
        inline void tan(const _T * pSrc, _T * pDst, int len)
        {
            for (int i = 0; i < len; ++i)
                pDst[i] = tan_cmath(pSrc[i]);
        }

        template<typename _T>
        inline void asin(const _T * pSrc, _T * pDst, int len)
        {
            for (int i = 0; i < len; ++i)
                pDst[i] = asin_cmath(pSrc[i]);
        }

        template<typename _T>
        inline void acos(const _T * pSrc, _T * pDst, int len)
        {
            for (int i = 0; i < len; ++i)
                pDst[i] = acos_cmath(pSrc[i]);
        }

        template<typename _T>
        inline void atan(const _T * pSrc, _T * pDst, int len)
        {
            for (int i = 0; i < len; ++i)
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
