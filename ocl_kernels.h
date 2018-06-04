#include <cstdint>

//int iGID = get_global_id(0);
//if (iGID >= size)
//    return;
#define OCL_IGUID_COMMON " int iGID = get_global_id(0); if (iGID >= size) return; "

namespace ocl
{
namespace internals
{
    /// There's some copy-paste here ;)
    struct Kernel
    {
        enum class Func : uint32_t
        {
            Add_32s,
            Sub_32s,
            Mul_32s,
            Div_32s,
            AddC_32s,
            SubC_32s,
            MulC_32s,
            DivC_32s,
            SubCRev_32s,
            DivCRev_32s,
            Abs_32s,
            //
            Add_32u,
            Sub_32u,
            Mul_32u,
            Div_32u,
            AddC_32u,
            SubC_32u,
            MulC_32u,
            DivC_32u,
            SubCRev_32u,
            DivCRev_32u,
            //
            Add_64s,
            Sub_64s,
            Mul_64s,
            Div_64s,
            AddC_64s,
            SubC_64s,
            MulC_64s,
            DivC_64s,
            SubCRev_64s,
            DivCRev_64s,
            Abs_64s,
            //
            Add_64u,
            Sub_64u,
            Mul_64u,
            Div_64u,
            AddC_64u,
            SubC_64u,
            MulC_64u,
            DivC_64u,
            SubCRev_64u,
            DivCRev_64u,
            //
            Add_32f,
            Sub_32f,
            Mul_32f,
            Div_32f,
            AddC_32f,
            SubC_32f,
            MulC_32f,
            DivC_32f,
            SubCRev_32f,
            DivCRev_32f,
            Abs_32f,
            //
            Add_64f,
            Sub_64f,
            Mul_64f,
            Div_64f,
            AddC_64f,
            SubC_64f,
            MulC_64f,
            DivC_64f,
            SubCRev_64f,
            DivCRev_64f,
            Abs_64f,
            Count
        };

        ///
        struct TextProgram
        {
            const char * name = nullptr;
            const char * text = nullptr;

            template <typename _T>
            static constexpr TextProgram create()
            {
                return {_T::programName(), _T::programText()};
            }
        };

        static constexpr uint32_t funcCount() { return (uint32_t)Func::Count; }

        template <typename _T>
        struct PtrPtrPtr
        {
            using ItemType = _T;
            using DataTypeSrc1 = const _T*;
            using DataTypeSrc2 = const _T*;
            using DataTypeDst = _T*;
            using BaseType = PtrPtrPtr<_T>;
        };

        template <typename _T>
        struct PtrValPtr
        {
            using ItemType = _T;
            using DataTypeSrc1 = const _T*;
            using DataTypeSrc2 = _T;
            using DataTypeDst = _T*;
            using BaseType = PtrValPtr<_T>;
        };

        // int32_t

        struct Add_32s : public PtrPtrPtr<int32_t>
        {
            static constexpr Func id() { return Func::Add_32s; }
            static constexpr const char * programName() { return "add_32s"; }
            static constexpr const char * programText()
            {
                return "__kernel void add_32s(__global const int* a, __global const int* b, __global int* c, int size) {"
                    OCL_IGUID_COMMON
                    "c[iGID] = a[iGID] + b[iGID]; }";
            }
        };

        struct Sub_32s : public PtrPtrPtr<int32_t>
        {
            static constexpr Func id() { return Func::Sub_32s; }
            static constexpr const char * programName() { return "sub_32s"; }
            static constexpr const char * programText()
            {
                return "__kernel void sub_32s(__global const int* a, __global const int* b, __global int* c, int size) {"
                    OCL_IGUID_COMMON
                    "c[iGID] = a[iGID] - b[iGID]; }";
            }
        };

        struct Mul_32s : public PtrPtrPtr<int32_t>
        {
            static constexpr Func id() { return Func::Mul_32s; }
            static constexpr const char * programName() { return "mul_32s"; }
            static constexpr const char * programText()
            {
                return " __kernel void mul_32s(__global const int* a, __global const int* b, __global int* c, int size) {"
                    OCL_IGUID_COMMON
                    "c[iGID] = a[iGID] * b[iGID]; }";
            }
        };

        struct Div_32s : public PtrPtrPtr<int32_t>
        {
            static constexpr Func id() { return Func::Div_32s; }
            static constexpr const char * programName() { return "div_32s"; }
            static constexpr const char * programText()
            {
                return "__kernel void div_32s(__global const int* a, __global const int* b, __global int* c, int size) {"
                    OCL_IGUID_COMMON
                    "c[iGID] = a[iGID] / b[iGID]; }";
            }
        };

        struct AddC_32s : public PtrValPtr<int32_t>
        {
            static constexpr Func id() { return Func::AddC_32s; }
            static constexpr const char * programName() { return "addC_32s"; }
            static constexpr const char * programText()
            {
                return "__kernel void addC_32s(__global const int* a, int b, __global int* c, int size) {"
                    OCL_IGUID_COMMON
                    "c[iGID] = a[iGID] + b; }";
            }
        };

        struct SubC_32s : public PtrValPtr<int32_t>
        {
            static constexpr Func id() { return Func::SubC_32s; }
            static constexpr const char * programName() { return "subC_32s"; }
            static constexpr const char * programText()
            {
                return "__kernel void subC_32s(__global const int* a, int b, __global int* c, int size) {"
                    OCL_IGUID_COMMON
                    "c[iGID] = a[iGID] - b; }";
            }
        };

        struct SubCRev_32s : public PtrValPtr<int32_t>
        {
            static constexpr Func id() { return Func::SubCRev_32s; }
            static constexpr const char * programName() { return "subCRev_32s"; }
            static constexpr const char * programText()
            {
                return "__kernel void subCRev_32s(__global const int* a, int b, __global int* c, int size) {"
                    OCL_IGUID_COMMON
                    "c[iGID] = b - a[iGID]; }";
            }
        };

        struct MulC_32s : public PtrValPtr<int32_t>
        {
            static constexpr Func id() { return Func::MulC_32s; }
            static constexpr const char * programName() { return "mulC_32s"; }
            static constexpr const char * programText()
            {
                return "__kernel void mulC_32s(__global const int* a, int b, __global int* c, int size) {"
                    OCL_IGUID_COMMON
                    "c[iGID] = a[iGID] * b; }";
            }
        };

        struct DivC_32s : public PtrValPtr<int32_t>
        {
            static constexpr Func id() { return Func::DivC_32s; }
            static constexpr const char * programName() { return "divC_32s"; }
            static constexpr const char * programText()
            {
                return "__kernel void divC_32s(__global const int* a, int b, __global int* c, int size) {"
                    OCL_IGUID_COMMON
                    "c[iGID] = a[iGID] / b; }";
            }
        };

        struct DivCRev_32s : public PtrValPtr<int32_t>
        {
            static constexpr Func id() { return Func::DivCRev_32s; }
            static constexpr const char * programName() { return "divCRev_32s"; }
            static constexpr const char * programText()
            {
                return "__kernel void divCRev_32s(__global const int* a, int b, __global int* c, int size) {"
                    OCL_IGUID_COMMON
                    "c[iGID] = b / a[iGID]; }";
            }
        };

        struct Abs_32s : public PtrValPtr<int32_t>
        {
            static constexpr Func id() { return Func::Abs_32s; }
            static constexpr const char * programName() { return "abs_32s"; }
            static constexpr const char * programText()
            {
                return R"(
                __kernel void abs_32s(__global const int* a, int b, __global int* c, int size)
                {
                    int iGID = get_global_id(0);
                    if (iGID >= size)
                        return;
                    if (a[iGID] < 0)
                        c[iGID] = -a[iGID];
                    else
                        c[iGID] = a[iGID];
                })";
            }
        };

        // uint32_t

        struct Add_32u : public PtrPtrPtr<uint32_t>
        {
            static constexpr Func id() { return Func::Add_32u; }
            static constexpr const char * programName() { return "add_32u"; }
            static constexpr const char * programText()
            {
                return "__kernel void add_32u(__global const uint* a, __global const uint* b, __global uint* c, int size) {"
                    OCL_IGUID_COMMON
                    "c[iGID] = a[iGID] + b[iGID]; }";
            }
        };

        struct Sub_32u : public PtrPtrPtr<uint32_t>
        {
            static constexpr Func id() { return Func::Sub_32u; }
            static constexpr const char * programName() { return "sub_32u"; }
            static constexpr const char * programText()
            {
                return "__kernel void sub_32u(__global const uint* a, __global const uint* b, __global uint* c, int size) {"
                    OCL_IGUID_COMMON
                    "c[iGID] = a[iGID] - b[iGID]; }";
            }
        };

        struct Mul_32u : public PtrPtrPtr<uint32_t>
        {
            static constexpr Func id() { return Func::Mul_32u; }
            static constexpr const char * programName() { return "mul_32u"; }
            static constexpr const char * programText()
            {
                return "__kernel void mul_32u(__global const uint* a, __global const uint* b, __global uint* c, int size) {"
                    OCL_IGUID_COMMON
                    "c[iGID] = a[iGID] * b[iGID]; }";
            }
        };

        struct Div_32u : public PtrPtrPtr<uint32_t>
        {
            static constexpr Func id() { return Func::Div_32u; }
            static constexpr const char * programName() { return "div_32u"; }
            static constexpr const char * programText()
            {
                return "__kernel void div_32u(__global const uint* a, __global const uint* b, __global uint* c, int size) {"
                    OCL_IGUID_COMMON
                    "c[iGID] = a[iGID] / b[iGID]; }";
            }
        };

        struct AddC_32u : public PtrValPtr<uint32_t>
        {
            static constexpr Func id() { return Func::AddC_32u; }
            static constexpr const char * programName() { return "addC_32u"; }
            static constexpr const char * programText()
            {
                return "__kernel void addC_32u(__global const uint* a, uint b, __global uint* c, int size) {"
                    OCL_IGUID_COMMON
                    "c[iGID] = a[iGID] + b; }";
            }
        };

        struct SubC_32u : public PtrValPtr<uint32_t>
        {
            static constexpr Func id() { return Func::SubC_32u; }
            static constexpr const char * programName() { return "subC_32u"; }
            static constexpr const char * programText()
            {
                return "__kernel void subC_32u(__global const uint* a, uint b, __global uint* c, int size) {"
                    OCL_IGUID_COMMON
                    "c[iGID] = a[iGID] - b; }";
            }
        };

        struct SubCRev_32u : public PtrValPtr<uint32_t>
        {
            static constexpr Func id() { return Func::SubCRev_32u; }
            static constexpr const char * programName() { return "subCRev_32u"; }
            static constexpr const char * programText()
            {
                return "__kernel void subCRev_32u(__global const uint* a, uint b, __global uint* c, int size) {"
                    OCL_IGUID_COMMON
                    "c[iGID] = b - a[iGID]; }";
            }
        };

        struct MulC_32u : public PtrValPtr<uint32_t>
        {
            static constexpr Func id() { return Func::MulC_32u; }
            static constexpr const char * programName() { return "mulC_32u"; }
            static constexpr const char * programText()
            {
                return "__kernel void mulC_32u(__global const uint* a, uint b, __global uint* c, int size) {"
                    OCL_IGUID_COMMON
                    "c[iGID] = a[iGID] * b; }";
            }
        };

        struct DivC_32u : public PtrValPtr<uint32_t>
        {
            static constexpr Func id() { return Func::DivC_32u; }
            static constexpr const char * programName() { return "divC_32u"; }
            static constexpr const char * programText()
            {
                return "__kernel void divC_32u(__global const uint* a, uint b, __global uint* c, int size) {"
                    OCL_IGUID_COMMON
                    "c[iGID] = a[iGID] / b; }";
            }
        };

        struct DivCRev_32u : public PtrValPtr<uint32_t>
        {
            static constexpr Func id() { return Func::DivCRev_32u; }
            static constexpr const char * programName() { return "divCRev_32u"; }
            static constexpr const char * programText()
            {
                return "__kernel void divCRev_32u(__global const uint* a, uint b, __global uint* c, int size) {"
                    OCL_IGUID_COMMON
                    "c[iGID] = b / a[iGID]; }";
            }
        };

        // int64_t

        struct Add_64s : public PtrPtrPtr<int64_t>
        {
            static constexpr Func id() { return Func::Add_64s; }
            static constexpr const char * programName() { return "add_64s"; }
            static constexpr const char * programText()
            {
                return "__kernel void add_64s(__global const long* a, __global const long* b, __global long* c, int size) {"
                    OCL_IGUID_COMMON
                    "c[iGID] = a[iGID] + b[iGID]; }";
            }
        };

        struct Sub_64s : public PtrPtrPtr<int64_t>
        {
            static constexpr Func id() { return Func::Sub_64s; }
            static constexpr const char * programName() { return "sub_64s"; }
            static constexpr const char * programText()
            {
                return "__kernel void sub_64s(__global const long* a, __global const long* b, __global long* c, int size) {"
                    OCL_IGUID_COMMON
                    "c[iGID] = a[iGID] - b[iGID]; }";
            }
        };

        struct Mul_64s : public PtrPtrPtr<int64_t>
        {
            static constexpr Func id() { return Func::Mul_64s; }
            static constexpr const char * programName() { return "mul_64s"; }
            static constexpr const char * programText()
            {
                return "__kernel void mul_64s(__global const long* a, __global const long* b, __global long* c, int size) {"
                    OCL_IGUID_COMMON
                    "c[iGID] = a[iGID] * b[iGID]; }";
            }
        };

        struct Div_64s : public PtrPtrPtr<int64_t>
        {
            static constexpr Func id() { return Func::Div_64s; }
            static constexpr const char * programName() { return "div_64s"; }
            static constexpr const char * programText()
            {
                return "__kernel void div_64s(__global const long* a, __global const long* b, __global long* c, int size) {"
                    OCL_IGUID_COMMON
                    "c[iGID] = a[iGID] / b[iGID]; }";
            }
        };

        struct AddC_64s : public PtrValPtr<int64_t>
        {
            static constexpr Func id() { return Func::AddC_64s; }
            static constexpr const char * programName() { return "addC_64s"; }
            static constexpr const char * programText()
            {
                return "__kernel void addC_64s(__global const long* a, long b, __global long* c, int size) {"
                    OCL_IGUID_COMMON
                    "c[iGID] = a[iGID] + b; }";
            }
        };

        struct SubC_64s : public PtrValPtr<int64_t>
        {
            static constexpr Func id() { return Func::SubC_64s; }
            static constexpr const char * programName() { return "subC_64s"; }
            static constexpr const char * programText()
            {
                return "__kernel void subC_64s(__global const long* a, long b, __global long* c, int size) {"
                    OCL_IGUID_COMMON
                    "c[iGID] = a[iGID] - b; }";
            }
        };

        struct SubCRev_64s : public PtrValPtr<int64_t>
        {
            static constexpr Func id() { return Func::SubCRev_64s; }
            static constexpr const char * programName() { return "subCRev_64s"; }
            static constexpr const char * programText()
            {
                return "__kernel void subCRev_64s(__global const long* a, long b, __global long* c, int size) {"
                    OCL_IGUID_COMMON
                    "c[iGID] = b - a[iGID]; }";
            }
        };

        struct MulC_64s : public PtrValPtr<int64_t>
        {
            static constexpr Func id() { return Func::MulC_64s; }
            static constexpr const char * programName() { return "mulC_64s"; }
            static constexpr const char * programText()
            {
                return "__kernel void mulC_64s(__global const long* a, long b, __global long* c, int size) {"
                    OCL_IGUID_COMMON
                    "c[iGID] = a[iGID] * b; }";
            }
        };

        struct DivC_64s : public PtrValPtr<int64_t>
        {
            static constexpr Func id() { return Func::DivC_64s; }
            static constexpr const char * programName() { return "divC_64s"; }
            static constexpr const char * programText()
            {
                return "__kernel void divC_64s(__global const long* a, long b, __global long* c, int size) {"
                    OCL_IGUID_COMMON
                    "c[iGID] = a[iGID] / b; }";
            }
        };

        struct DivCRev_64s : public PtrValPtr<int64_t>
        {
            static constexpr Func id() { return Func::DivCRev_64s; }
            static constexpr const char * programName() { return "divCRev_64s"; }
            static constexpr const char * programText()
            {
                return "__kernel void divCRev_64s(__global const long* a, long b, __global long* c, int size) {"
                    OCL_IGUID_COMMON
                    "c[iGID] = b / a[iGID]; }";
            }
        };

        struct Abs_64s : public PtrValPtr<int64_t>
        {
            static constexpr Func id() { return Func::Abs_64s; }
            static constexpr const char * programName() { return "abs_64s"; }
            static constexpr const char * programText()
            {
                return R"(
                __kernel void abs_64s(__global const long* a, long b, __global long* c, int size)
                {
                    int iGID = get_global_id(0);
                    if (iGID >= size)
                        return;
                    if (a[iGID] < 0)
                        c[iGID] = -a[iGID];
                    else
                        c[iGID] = a[iGID];
                })";
            }
        };

        // uint64_t

        struct Add_64u : public PtrPtrPtr<uint64_t>
        {
            static constexpr Func id() { return Func::Add_64u; }
            static constexpr const char * programName() { return "add_64u"; }
            static constexpr const char * programText()
            {
                return "__kernel void add_64u(__global const ulong* a, __global const ulong* b, __global ulong* c, int size) {"
                    OCL_IGUID_COMMON
                    "c[iGID] = a[iGID] + b[iGID]; }";
            }
        };

        struct Sub_64u : public PtrPtrPtr<uint64_t>
        {
            static constexpr Func id() { return Func::Sub_64u; }
            static constexpr const char * programName() { return "sub_64u"; }
            static constexpr const char * programText()
            {
                return "__kernel void sub_64u(__global const ulong* a, __global const ulong* b, __global ulong* c, int size) {"
                    OCL_IGUID_COMMON
                    "c[iGID] = a[iGID] - b[iGID]; }";
            }
        };

        struct Mul_64u : public PtrPtrPtr<uint64_t>
        {
            static constexpr Func id() { return Func::Mul_64u; }
            static constexpr const char * programName() { return "mul_64u"; }
            static constexpr const char * programText()
            {
                return "__kernel void mul_64u(__global const ulong* a, __global const ulong* b, __global ulong* c, int size) {"
                    OCL_IGUID_COMMON
                    "c[iGID] = a[iGID] * b[iGID]; }";
            }
        };

        struct Div_64u : public PtrPtrPtr<uint64_t>
        {
            static constexpr Func id() { return Func::Div_64u; }
            static constexpr const char * programName() { return "div_64u"; }
            static constexpr const char * programText()
            {
                return "__kernel void div_64u(__global const ulong* a, __global const ulong* b, __global ulong* c, int size) {"
                    OCL_IGUID_COMMON
                    "c[iGID] = a[iGID] / b[iGID]; }";
            }
        };

        struct AddC_64u : public PtrValPtr<uint64_t>
        {
            static constexpr Func id() { return Func::AddC_64u; }
            static constexpr const char * programName() { return "addC_64u"; }
            static constexpr const char * programText()
            {
                return "__kernel void addC_64u(__global const ulong* a, ulong b, __global ulong* c, int size) {"
                    OCL_IGUID_COMMON
                    "c[iGID] = a[iGID] + b; }";
            }
        };

        struct SubC_64u : public PtrValPtr<uint64_t>
        {
            static constexpr Func id() { return Func::SubC_64u; }
            static constexpr const char * programName() { return "subC_64u"; }
            static constexpr const char * programText()
            {
                return "__kernel void subC_64u(__global const ulong* a, ulong b, __global ulong* c, int size) {"
                    OCL_IGUID_COMMON
                    "c[iGID] = a[iGID] - b; }";
            }
        };

        struct SubCRev_64u : public PtrValPtr<uint64_t>
        {
            static constexpr Func id() { return Func::SubCRev_64u; }
            static constexpr const char * programName() { return "subCRev_64u"; }
            static constexpr const char * programText()
            {
                return "__kernel void subCRev_64u(__global const ulong* a, ulong b, __global ulong* c, int size) {"
                    OCL_IGUID_COMMON
                    "c[iGID] = b - a[iGID]; }";
            }
        };

        struct MulC_64u : public PtrValPtr<uint64_t>
        {
            static constexpr Func id() { return Func::MulC_64u; }
            static constexpr const char * programName() { return "mulC_64u"; }
            static constexpr const char * programText()
            {
                return "__kernel void mulC_64u(__global const ulong* a, ulong b, __global ulong* c, int size) {"
                    OCL_IGUID_COMMON
                    "c[iGID] = a[iGID] * b; }";
            }
        };

        struct DivC_64u : public PtrValPtr<uint64_t>
        {
            static constexpr Func id() { return Func::DivC_64u; }
            static constexpr const char * programName() { return "divC_64u"; }
            static constexpr const char * programText()
            {
                return "__kernel void divC_64u(__global const ulong* a, ulong b, __global ulong* c, int size) {"
                    OCL_IGUID_COMMON
                    "c[iGID] = a[iGID] / b; }";
            }
        };

        struct DivCRev_64u : public PtrValPtr<uint64_t>
        {
            static constexpr Func id() { return Func::DivCRev_64u; }
            static constexpr const char * programName() { return "divCRev_64u"; }
            static constexpr const char * programText()
            {
                return "__kernel void divCRev_64u(__global const ulong* a, ulong b, __global ulong* c, int size) {"
                    OCL_IGUID_COMMON
                    "c[iGID] = b / a[iGID]; }";
            }
        };

        // float

        struct Add_32f : public PtrPtrPtr<float>
        {
            static constexpr Func id() { return Func::Add_32f; }
            static constexpr const char * programName() { return "add_32f"; }
            static constexpr const char * programText()
            {
                return "__kernel void add_32f(__global const float* a, __global const float* b, __global float* c, int size) {"
                    OCL_IGUID_COMMON
                    "c[iGID] = a[iGID] + b[iGID]; }";
            }
        };

        struct Sub_32f : public PtrPtrPtr<float>
        {
            static constexpr Func id() { return Func::Sub_32f; }
            static constexpr const char * programName() { return "sub_32f"; }
            static constexpr const char * programText()
            {
                return "__kernel void sub_32f(__global const float* a, __global const float* b, __global float* c, int size) {"
                    OCL_IGUID_COMMON
                    "c[iGID] = a[iGID] - b[iGID]; }";
            }
        };

        struct Mul_32f : public PtrPtrPtr<float>
        {
            static constexpr Func id() { return Func::Mul_32f; }
            static constexpr const char * programName() { return "mul_32f"; }
            static constexpr const char * programText()
            {
                return "__kernel void mul_32f(__global const float* a, __global const float* b, __global float* c, int size) {"
                    OCL_IGUID_COMMON
                    "c[iGID] = a[iGID] * b[iGID]; }";
            }
        };

        struct Div_32f : public PtrPtrPtr<float>
        {
            static constexpr Func id() { return Func::Div_32f; }
            static constexpr const char * programName() { return "div_32f"; }
            static constexpr const char * programText()
            {
                return "__kernel void div_32f(__global const float* a, __global const float* b, __global float* c, int size) {"
                    OCL_IGUID_COMMON
                    "c[iGID] = a[iGID] / b[iGID]; }";
            }
        };

        struct AddC_32f : public PtrValPtr<float>
        {
            static constexpr Func id() { return Func::AddC_32f; }
            static constexpr const char * programName() { return "addC_32f"; }
            static constexpr const char * programText()
            {
                return "__kernel void addC_32f(__global const float* a, float b, __global float* c, int size) {"
                    OCL_IGUID_COMMON
                    "c[iGID] = a[iGID] + b; }";
            }
        };

        struct SubC_32f : public PtrValPtr<float>
        {
            static constexpr Func id() { return Func::SubC_32f; }
            static constexpr const char * programName() { return "subC_32f"; }
            static constexpr const char * programText()
            {
                return "__kernel void subC_32f(__global const float* a, float b, __global float* c, int size) {"
                    OCL_IGUID_COMMON
                    "c[iGID] = a[iGID] - b; }";
            }
        };

        struct SubCRev_32f : public PtrValPtr<float>
        {
            static constexpr Func id() { return Func::SubCRev_32f; }
            static constexpr const char * programName() { return "subCRev_32f"; }
            static constexpr const char * programText()
            {
                return "__kernel void subCRev_32f(__global const float* a, float b, __global float* c, int size) {"
                    OCL_IGUID_COMMON
                    "c[iGID] = b - a[iGID]; }";
            }
        };

        struct MulC_32f : public PtrValPtr<float>
        {
            static constexpr Func id() { return Func::MulC_32f; }
            static constexpr const char * programName() { return "mulC_32f"; }
            static constexpr const char * programText()
            {
                return "__kernel void mulC_32f(__global const float* a, float b, __global float* c, int size) {"
                    OCL_IGUID_COMMON
                    "c[iGID] = a[iGID] * b; }";
            }
        };

        struct DivC_32f : public PtrValPtr<float>
        {
            static constexpr Func id() { return Func::DivC_32f; }
            static constexpr const char * programName() { return "divC_32f"; }
            static constexpr const char * programText()
            {
                return "__kernel void divC_32f(__global const float* a, float b, __global float* c, int size) {"
                    OCL_IGUID_COMMON
                    "c[iGID] = a[iGID] / b; }";
            }
        };

        struct DivCRev_32f : public PtrValPtr<float>
        {
            static constexpr Func id() { return Func::DivCRev_32f; }
            static constexpr const char * programName() { return "divCRev_32f"; }
            static constexpr const char * programText()
            {
                return "__kernel void divCRev_32f(__global const float* a, float b, __global float* c, int size) {"
                    OCL_IGUID_COMMON
                    "c[iGID] = b / a[iGID]; }";
            }
        };

        struct Abs_32f : public PtrValPtr<float>
        {
            static constexpr Func id() { return Func::Abs_32f; }
            static constexpr const char * programName() { return "abs_32f"; }
            static constexpr const char * programText()
            {
                return R"(
                __kernel void abs_32f(__global const float* a, float b, __global float* c, int size)
                {
                    int iGID = get_global_id(0);
                    if (iGID >= size)
                        return;
                    if (a[iGID] < 0)
                        c[iGID] = -a[iGID];
                    else
                        c[iGID] = a[iGID];
                })";
            }
        };

        // double

        struct Add_64f : public PtrPtrPtr<double>
        {
            static constexpr Func id() { return Func::Add_64f; }
            static constexpr const char * programName() { return "add_64f"; }
            static constexpr const char * programText()
            {
                return "__kernel void add_64f(__global const double* a, __global const double* b, __global double* c, int size) {"
                    OCL_IGUID_COMMON
                    "c[iGID] = a[iGID] + b[iGID]; }";
            }
        };

        struct Sub_64f : public PtrPtrPtr<double>
        {
            static constexpr Func id() { return Func::Sub_64f; }
            static constexpr const char * programName() { return "sub_64f"; }
            static constexpr const char * programText()
            {
                return "__kernel void sub_64f(__global const double* a, __global const double* b, __global double* c, int size) {"
                    OCL_IGUID_COMMON
                    "c[iGID] = a[iGID] - b[iGID]; }";
            }
        };

        struct Mul_64f : public PtrPtrPtr<double>
        {
            static constexpr Func id() { return Func::Mul_64f; }
            static constexpr const char * programName() { return "mul_64f"; }
            static constexpr const char * programText()
            {
                return "__kernel void mul_64f(__global const double* a, __global const double* b, __global double* c, int size) {"
                    OCL_IGUID_COMMON
                    "c[iGID] = a[iGID] * b[iGID]; }";
            }
        };

        struct Div_64f : public PtrPtrPtr<double>
        {
            static constexpr Func id() { return Func::Div_64f; }
            static constexpr const char * programName() { return "div_64f"; }
            static constexpr const char * programText()
            {
                return "__kernel void div_64f(__global const double* a, __global const double* b, __global double* c, int size) {"
                    OCL_IGUID_COMMON
                    "c[iGID] = a[iGID] / b[iGID]; }";
            }
        };

        struct AddC_64f : public PtrValPtr<double>
        {
            static constexpr Func id() { return Func::AddC_64f; }
            static constexpr const char * programName() { return "addC_64f"; }
            static constexpr const char * programText()
            {
                return "__kernel void addC_64f(__global const double* a, double b, __global double* c, int size) {"
                    OCL_IGUID_COMMON
                    "c[iGID] = a[iGID] + b; }";
            }
        };

        struct SubC_64f : public PtrValPtr<double>
        {
            static constexpr Func id() { return Func::SubC_64f; }
            static constexpr const char * programName() { return "subC_64f"; }
            static constexpr const char * programText()
            {
                return "__kernel void subC_64f(__global const double* a, double b, __global double* c, int size) {"
                    OCL_IGUID_COMMON
                    "c[iGID] = a[iGID] - b; }";
            }
        };

        struct SubCRev_64f : public PtrValPtr<double>
        {
            static constexpr Func id() { return Func::SubCRev_64f; }
            static constexpr const char * programName() { return "subCRev_64f"; }
            static constexpr const char * programText()
            {
                return "__kernel void subCRev_64f(__global const double* a, double b, __global double* c, int size) {"
                    OCL_IGUID_COMMON
                    "c[iGID] = b - a[iGID]; }";
            }
        };

        struct MulC_64f : public PtrValPtr<double>
        {
            static constexpr Func id() { return Func::MulC_64f; }
            static constexpr const char * programName() { return "mulC_64f"; }
            static constexpr const char * programText()
            {
                return "__kernel void mulC_64f(__global const double* a, double b, __global double* c, int size) {"
                    OCL_IGUID_COMMON
                    "c[iGID] = a[iGID] * b; }";
            }
        };

        struct DivC_64f : public PtrValPtr<double>
        {
            static constexpr Func id() { return Func::DivC_64f; }
            static constexpr const char * programName() { return "divC_64f"; }
            static constexpr const char * programText()
            {
                return "__kernel void divC_64f(__global const double* a, double b, __global double* c, int size) {"
                    OCL_IGUID_COMMON
                    "c[iGID] = a[iGID] / b; }";
            }
        };

        struct DivCRev_64f : public PtrValPtr<double>
        {
            static constexpr Func id() { return Func::DivCRev_64f; }
            static constexpr const char * programName() { return "divCRev_64f"; }
            static constexpr const char * programText()
            {
                return "__kernel void divCRev_64f(__global const double* a, double b, __global double* c, int size) {"
                    OCL_IGUID_COMMON
                    "c[iGID] = b / a[iGID]; }";
            }
        };

        struct Abs_64f : public PtrValPtr<double>
        {
            static constexpr Func id() { return Func::Abs_64f; }
            static constexpr const char * programName() { return "abs_64f"; }
            static constexpr const char * programText()
            {
                return R"(
                __kernel void abs_64f(__global const double* a, double b, __global double* c, int size)
                {
                    int iGID = get_global_id(0);
                    if (iGID >= size)
                        return;
                    if (a[iGID] < 0)
                        c[iGID] = -a[iGID];
                    else
                        c[iGID] = a[iGID];
                })";
            }
        };

        //

        static constexpr TextProgram program(Func func)
        {
            switch (func) {
                case Func::Add_32s: return TextProgram::create<Add_32s>();
                case Func::Sub_32s: return TextProgram::create<Sub_32s>();
                case Func::Mul_32s: return TextProgram::create<Mul_32s>();
                case Func::Div_32s: return TextProgram::create<Div_32s>();
                case Func::AddC_32s: return TextProgram::create<AddC_32s>();
                case Func::SubC_32s: return TextProgram::create<SubC_32s>();
                case Func::MulC_32s: return TextProgram::create<MulC_32s>();
                case Func::DivC_32s: return TextProgram::create<DivC_32s>();
                case Func::SubCRev_32s: return TextProgram::create<SubCRev_32s>();
                case Func::DivCRev_32s: return TextProgram::create<DivCRev_32s>();
                //
                case Func::Add_32u: return TextProgram::create<Add_32u>();
                case Func::Sub_32u: return TextProgram::create<Sub_32u>();
                case Func::Mul_32u: return TextProgram::create<Mul_32u>();
                case Func::Div_32u: return TextProgram::create<Div_32u>();
                case Func::AddC_32u: return TextProgram::create<AddC_32u>();
                case Func::SubC_32u: return TextProgram::create<SubC_32u>();
                case Func::MulC_32u: return TextProgram::create<MulC_32u>();
                case Func::DivC_32u: return TextProgram::create<DivC_32u>();
                case Func::SubCRev_32u: return TextProgram::create<SubCRev_32u>();
                case Func::DivCRev_32u: return TextProgram::create<DivCRev_32u>();
                //
                case Func::Add_64s: return TextProgram::create<Add_64s>();
                case Func::Sub_64s: return TextProgram::create<Sub_64s>();
                case Func::Mul_64s: return TextProgram::create<Mul_64s>();
                case Func::Div_64s: return TextProgram::create<Div_64s>();
                case Func::AddC_64s: return TextProgram::create<AddC_64s>();
                case Func::SubC_64s: return TextProgram::create<SubC_64s>();
                case Func::MulC_64s: return TextProgram::create<MulC_64s>();
                case Func::DivC_64s: return TextProgram::create<DivC_64s>();
                case Func::SubCRev_64s: return TextProgram::create<SubCRev_64s>();
                case Func::DivCRev_64s: return TextProgram::create<DivCRev_64s>();
                //
                case Func::Add_64u: return TextProgram::create<Add_64u>();
                case Func::Sub_64u: return TextProgram::create<Sub_64u>();
                case Func::Mul_64u: return TextProgram::create<Mul_64u>();
                case Func::Div_64u: return TextProgram::create<Div_64u>();
                case Func::AddC_64u: return TextProgram::create<AddC_64u>();
                case Func::SubC_64u: return TextProgram::create<SubC_64u>();
                case Func::MulC_64u: return TextProgram::create<MulC_64u>();
                case Func::DivC_64u: return TextProgram::create<DivC_64u>();
                case Func::SubCRev_64u: return TextProgram::create<SubCRev_64u>();
                case Func::DivCRev_64u: return TextProgram::create<DivCRev_64u>();
                //
                case Func::Add_32f: return TextProgram::create<Add_32f>();
                case Func::Sub_32f: return TextProgram::create<Sub_32f>();
                case Func::Mul_32f: return TextProgram::create<Mul_32f>();
                case Func::Div_32f: return TextProgram::create<Div_32f>();
                case Func::AddC_32f: return TextProgram::create<AddC_32f>();
                case Func::SubC_32f: return TextProgram::create<SubC_32f>();
                case Func::MulC_32f: return TextProgram::create<MulC_32f>();
                case Func::DivC_32f: return TextProgram::create<DivC_32f>();
                case Func::SubCRev_32f: return TextProgram::create<SubCRev_32f>();
                case Func::DivCRev_32f: return TextProgram::create<DivCRev_32f>();
                //
                case Func::Add_64f: return TextProgram::create<Add_64f>();
                case Func::Sub_64f: return TextProgram::create<Sub_64f>();
                case Func::Mul_64f: return TextProgram::create<Mul_64f>();
                case Func::Div_64f: return TextProgram::create<Div_64f>();
                case Func::AddC_64f: return TextProgram::create<AddC_64f>();
                case Func::SubC_64f: return TextProgram::create<SubC_64f>();
                case Func::MulC_64f: return TextProgram::create<MulC_64f>();
                case Func::DivC_64f: return TextProgram::create<DivC_64f>();
                case Func::SubCRev_64f: return TextProgram::create<SubCRev_64f>();
                case Func::DivCRev_64f: return TextProgram::create<DivCRev_64f>();
                //
                //
                case Func::Abs_32s: return TextProgram::create<Abs_32s>();
                case Func::Abs_64s: return TextProgram::create<Abs_64s>();
                case Func::Abs_32f: return TextProgram::create<Abs_32f>();
                case Func::Abs_64f: return TextProgram::create<Abs_64f>();
                case Func::Count:
                    break;
            }
            return {nullptr, nullptr};
        }
    };
} // internals
} // ocl
