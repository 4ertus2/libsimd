#pragma once
#include <ippcore.h>

namespace ipp
{
    class IppException
    {
    public:
        IppException(IppStatus s, const char * fn)
        :	status_(s), function_(fn)
        {}

        IppStatus status() const { return status_; }

        const char * what() const { return ippGetStatusString(status_); }
        const char * func() const { return function_; }

    private:
        IppStatus status_;
        const char * function_;
    };
}

#ifdef SEGFAULT_ON_IPP_ERROR
#define STATUS_CHECK(x) {IppStatus s=(x); if(s != ippStsNoErr) {int* p=0; *p = *p;} }
#else
#define STATUS_CHECK(x) {IppStatus s=(x); if(s != ippStsNoErr) throw ipp::IppException(s, __FUNCTION__);}
#endif
