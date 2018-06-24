#pragma once
#include <cstdint>
#include <string>

namespace ocl
{
namespace internals
{
namespace Kernel
{
    enum class Func : uint32_t
    {
        Add_8s,
        Sub_8s,
        Mul_8s,
        Div_8s,
        AddC_8s,
        SubC_8s,
        MulC_8s,
        DivC_8s,
        SubCRev_8s,
        DivCRev_8s,
        Abs_8s,
        //
        Add_8u,
        Sub_8u,
        Mul_8u,
        Div_8u,
        AddC_8u,
        SubC_8u,
        MulC_8u,
        DivC_8u,
        SubCRev_8u,
        DivCRev_8u,
        //
        Add_16s,
        Sub_16s,
        Mul_16s,
        Div_16s,
        AddC_16s,
        SubC_16s,
        MulC_16s,
        DivC_16s,
        SubCRev_16s,
        DivCRev_16s,
        Abs_16s,
        //
        Add_16u,
        Sub_16u,
        Mul_16u,
        Div_16u,
        AddC_16u,
        SubC_16u,
        MulC_16u,
        DivC_16u,
        SubCRev_16u,
        DivCRev_16u,
        //
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
        //
        Count
    };

    inline constexpr uint32_t funcCount() { return (uint32_t)Func::Count; }

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

        template <typename _T>
        static constexpr const char * typeStr();

        template <typename _T1, typename _T2, typename _T3>
        static std::string func3args(const char * func, const char * operation)
        {
            static const char * funcTemplate =
                "__kernel void %s(%s a, %s b, %s c, int size) {"
                " int iGID = get_global_id(0);"
                " if (iGID >= size) return;"
                " %s; }";
            char buf[1024];
            snprintf(buf, 1024, funcTemplate, func, typeStr<_T1>(), typeStr<_T2>(), typeStr<_T3>(), operation);
            return std::string(buf);
        }
    };

    template <> constexpr const char * TextProgram::typeStr<int8_t>() { return "char"; }
    template <> constexpr const char * TextProgram::typeStr<uint8_t>() { return "uchar"; }
    template <> constexpr const char * TextProgram::typeStr<int16_t>() { return "short"; }
    template <> constexpr const char * TextProgram::typeStr<uint16_t>() { return "ushort"; }
    template <> constexpr const char * TextProgram::typeStr<int32_t>() { return "int"; }
    template <> constexpr const char * TextProgram::typeStr<uint32_t>() { return "uint"; }
    template <> constexpr const char * TextProgram::typeStr<int64_t>() { return "long"; }
    template <> constexpr const char * TextProgram::typeStr<uint64_t>() { return "ulong"; }
    template <> constexpr const char * TextProgram::typeStr<float>() { return "float"; }
    template <> constexpr const char * TextProgram::typeStr<double>() { return "double"; }
    template <> constexpr const char * TextProgram::typeStr<int8_t*>() { return "__global char*"; }
    template <> constexpr const char * TextProgram::typeStr<uint8_t*>() { return "__global uchar*"; }
    template <> constexpr const char * TextProgram::typeStr<int16_t*>() { return "__global short*"; }
    template <> constexpr const char * TextProgram::typeStr<uint16_t*>() { return "__global ushort*"; }
    template <> constexpr const char * TextProgram::typeStr<int32_t*>() { return "__global int*"; }
    template <> constexpr const char * TextProgram::typeStr<uint32_t*>() { return "__global uint*"; }
    template <> constexpr const char * TextProgram::typeStr<int64_t*>() { return "__global long*"; }
    template <> constexpr const char * TextProgram::typeStr<uint64_t*>() { return "__global ulong*"; }
    template <> constexpr const char * TextProgram::typeStr<float*>() { return "__global float*"; }
    template <> constexpr const char * TextProgram::typeStr<double*>() { return "__global double*"; }
    template <> constexpr const char * TextProgram::typeStr<const int8_t*>() { return "__global const char*"; }
    template <> constexpr const char * TextProgram::typeStr<const uint8_t*>() { return "__global const uchar*"; }
    template <> constexpr const char * TextProgram::typeStr<const int16_t*>() { return "__global const short*"; }
    template <> constexpr const char * TextProgram::typeStr<const uint16_t*>() { return "__global const ushort*"; }
    template <> constexpr const char * TextProgram::typeStr<const int32_t*>() { return "__global const int*"; }
    template <> constexpr const char * TextProgram::typeStr<const uint32_t*>() { return "__global const uint*"; }
    template <> constexpr const char * TextProgram::typeStr<const int64_t*>() { return "__global const long*"; }
    template <> constexpr const char * TextProgram::typeStr<const uint64_t*>() { return "__global const ulong*"; }
    template <> constexpr const char * TextProgram::typeStr<const float*>() { return "__global const float*"; }
    template <> constexpr const char * TextProgram::typeStr<const double*>() { return "__global const double*"; }

    template <typename _T>
    struct PtrPtrPtr
    {
        using ItemType = _T;
        using DataTypeSrc1 = const _T*;
        using DataTypeSrc2 = const _T*;
        using DataTypeDst = _T*;
        using BaseType = PtrPtrPtr<_T>;

        static constexpr const char * opAdd() { return "c[iGID] = a[iGID] + b[iGID]"; }
        static constexpr const char * opSub() { return "c[iGID] = a[iGID] - b[iGID]"; }
        static constexpr const char * opMul() { return "c[iGID] = a[iGID] * b[iGID]"; }
        static constexpr const char * opDiv() { return "c[iGID] = a[iGID] / b[iGID]"; }
    };

    template <typename _T>
    struct PtrValPtr
    {
        using ItemType = _T;
        using DataTypeSrc1 = const _T*;
        using DataTypeSrc2 = _T;
        using DataTypeDst = _T*;
        using BaseType = PtrValPtr<_T>;

        static constexpr const char * opAddC() { return "c[iGID] = a[iGID] + b"; }
        static constexpr const char * opSubC() { return "c[iGID] = a[iGID] - b"; }
        static constexpr const char * opMulC() { return "c[iGID] = a[iGID] * b"; }
        static constexpr const char * opDivC() { return "c[iGID] = a[iGID] / b"; }
        static constexpr const char * opSubCRev() { return "c[iGID] = b - a[iGID]"; }
        static constexpr const char * opDivCRev() { return "c[iGID] = b / a[iGID]"; }

        static constexpr const char * opAbs()
        {
            return "if (a[iGID] < 0) c[iGID] = -a[iGID]; else c[iGID] = a[iGID]";
        }
    };

    //

    template <typename _T>
    struct Add : public PtrPtrPtr<_T>
    {
        static constexpr Func id();
        static constexpr const char * programName();
        static const char * programText()
        {
            static std::string text = TextProgram::func3args<const _T*, const _T*, _T*>(programName(), PtrPtrPtr<_T>::opAdd());
            return text.data();
        }
    };

    template <> constexpr Func Add<int8_t>::id() { return Func::Add_8s; }
    template <> constexpr Func Add<uint8_t>::id() { return Func::Add_8u; }
    template <> constexpr Func Add<int16_t>::id() { return Func::Add_16s; }
    template <> constexpr Func Add<uint16_t>::id() { return Func::Add_16u; }
    template <> constexpr Func Add<int32_t>::id() { return Func::Add_32s; }
    template <> constexpr Func Add<uint32_t>::id() { return Func::Add_32u; }
    template <> constexpr Func Add<int64_t>::id() { return Func::Add_64s; }
    template <> constexpr Func Add<uint64_t>::id() { return Func::Add_64u; }
    template <> constexpr Func Add<float>::id() { return Func::Add_32f; }
    template <> constexpr Func Add<double>::id() { return Func::Add_64f; }
    template <> constexpr const char * Add<int8_t>::programName() { return "add_8s"; }
    template <> constexpr const char * Add<uint8_t>::programName() { return "add_8u"; }
    template <> constexpr const char * Add<int16_t>::programName() { return "add_16s"; }
    template <> constexpr const char * Add<uint16_t>::programName() { return "add_16u"; }
    template <> constexpr const char * Add<int32_t>::programName() { return "add_32s"; }
    template <> constexpr const char * Add<uint32_t>::programName() { return "add_32u"; }
    template <> constexpr const char * Add<int64_t>::programName() { return "add_64s"; }
    template <> constexpr const char * Add<uint64_t>::programName() { return "add_64u"; }
    template <> constexpr const char * Add<float>::programName() { return "add_32f"; }
    template <> constexpr const char * Add<double>::programName() { return "add_64f"; }

    template <typename _T>
    struct Sub : public PtrPtrPtr<_T>
    {
        static constexpr Func id();
        static constexpr const char * programName();
        static const char * programText()
        {
            static std::string text = TextProgram::func3args<const _T*, const _T*, _T*>(programName(), PtrPtrPtr<_T>::opSub());
            return text.data();
        }
    };

    template <> constexpr Func Sub<int8_t>::id() { return Func::Sub_8s; }
    template <> constexpr Func Sub<uint8_t>::id() { return Func::Sub_8u; }
    template <> constexpr Func Sub<int16_t>::id() { return Func::Sub_16s; }
    template <> constexpr Func Sub<uint16_t>::id() { return Func::Sub_16u; }
    template <> constexpr Func Sub<int32_t>::id() { return Func::Sub_32s; }
    template <> constexpr Func Sub<uint32_t>::id() { return Func::Sub_32u; }
    template <> constexpr Func Sub<int64_t>::id() { return Func::Sub_64s; }
    template <> constexpr Func Sub<uint64_t>::id() { return Func::Sub_64u; }
    template <> constexpr Func Sub<float>::id() { return Func::Sub_32f; }
    template <> constexpr Func Sub<double>::id() { return Func::Sub_64f; }
    template <> constexpr const char * Sub<int8_t>::programName() { return "sub_8s"; }
    template <> constexpr const char * Sub<uint8_t>::programName() { return "sub_8u"; }
    template <> constexpr const char * Sub<int16_t>::programName() { return "sub_16s"; }
    template <> constexpr const char * Sub<uint16_t>::programName() { return "sub_16u"; }
    template <> constexpr const char * Sub<int32_t>::programName() { return "sub_32s"; }
    template <> constexpr const char * Sub<uint32_t>::programName() { return "sub_32u"; }
    template <> constexpr const char * Sub<int64_t>::programName() { return "sub_64s"; }
    template <> constexpr const char * Sub<uint64_t>::programName() { return "sub_64u"; }
    template <> constexpr const char * Sub<float>::programName() { return "sub_32f"; }
    template <> constexpr const char * Sub<double>::programName() { return "sub_64f"; }

    template <typename _T>
    struct Mul : public PtrPtrPtr<_T>
    {
        static constexpr Func id();
        static constexpr const char * programName();
        static const char * programText()
        {
            static std::string text = TextProgram::func3args<const _T*, const _T*, _T*>(programName(), PtrPtrPtr<_T>::opMul());
            return text.data();
        }
    };

    template <> constexpr Func Mul<int8_t>::id() { return Func::Mul_8s; }
    template <> constexpr Func Mul<uint8_t>::id() { return Func::Mul_8u; }
    template <> constexpr Func Mul<int16_t>::id() { return Func::Mul_16s; }
    template <> constexpr Func Mul<uint16_t>::id() { return Func::Mul_16u; }
    template <> constexpr Func Mul<int32_t>::id() { return Func::Mul_32s; }
    template <> constexpr Func Mul<uint32_t>::id() { return Func::Mul_32u; }
    template <> constexpr Func Mul<int64_t>::id() { return Func::Mul_64s; }
    template <> constexpr Func Mul<uint64_t>::id() { return Func::Mul_64u; }
    template <> constexpr Func Mul<float>::id() { return Func::Mul_32f; }
    template <> constexpr Func Mul<double>::id() { return Func::Mul_64f; }
    template <> constexpr const char * Mul<int8_t>::programName() { return "mul_8s"; }
    template <> constexpr const char * Mul<uint8_t>::programName() { return "mul_8u"; }
    template <> constexpr const char * Mul<int16_t>::programName() { return "mul_16s"; }
    template <> constexpr const char * Mul<uint16_t>::programName() { return "mul_16u"; }
    template <> constexpr const char * Mul<int32_t>::programName() { return "mul_32s"; }
    template <> constexpr const char * Mul<uint32_t>::programName() { return "mul_32u"; }
    template <> constexpr const char * Mul<int64_t>::programName() { return "mul_64s"; }
    template <> constexpr const char * Mul<uint64_t>::programName() { return "mul_64u"; }
    template <> constexpr const char * Mul<float>::programName() { return "mul_32f"; }
    template <> constexpr const char * Mul<double>::programName() { return "mul_64f"; }

    template <typename _T>
    struct Div : public PtrPtrPtr<_T>
    {
        static constexpr Func id();
        static constexpr const char * programName();
        static const char * programText()
        {
            static std::string text = TextProgram::func3args<const _T*, const _T*, _T*>(programName(), PtrPtrPtr<_T>::opDiv());
            return text.data();
        }
    };

    template <> constexpr Func Div<int8_t>::id() { return Func::Div_8s; }
    template <> constexpr Func Div<uint8_t>::id() { return Func::Div_8u; }
    template <> constexpr Func Div<int16_t>::id() { return Func::Div_16s; }
    template <> constexpr Func Div<uint16_t>::id() { return Func::Div_16u; }
    template <> constexpr Func Div<int32_t>::id() { return Func::Div_32s; }
    template <> constexpr Func Div<uint32_t>::id() { return Func::Div_32u; }
    template <> constexpr Func Div<int64_t>::id() { return Func::Div_64s; }
    template <> constexpr Func Div<uint64_t>::id() { return Func::Div_64u; }
    template <> constexpr Func Div<float>::id() { return Func::Div_32f; }
    template <> constexpr Func Div<double>::id() { return Func::Div_64f; }
    template <> constexpr const char * Div<int8_t>::programName() { return "div_8s"; }
    template <> constexpr const char * Div<uint8_t>::programName() { return "div_8u"; }
    template <> constexpr const char * Div<int16_t>::programName() { return "div_16s"; }
    template <> constexpr const char * Div<uint16_t>::programName() { return "div_16u"; }
    template <> constexpr const char * Div<int32_t>::programName() { return "div_32s"; }
    template <> constexpr const char * Div<uint32_t>::programName() { return "div_32u"; }
    template <> constexpr const char * Div<int64_t>::programName() { return "div_64s"; }
    template <> constexpr const char * Div<uint64_t>::programName() { return "div_64u"; }
    template <> constexpr const char * Div<float>::programName() { return "div_32f"; }
    template <> constexpr const char * Div<double>::programName() { return "div_64f"; }

    template <typename _T>
    struct AddC : public PtrValPtr<_T>
    {
        static constexpr Func id();
        static constexpr const char * programName();
        static const char * programText()
        {
            static std::string text = TextProgram::func3args<const _T*, _T, _T*>(programName(), PtrValPtr<_T>::opAddC());
            return text.data();
        }
    };

    template <> constexpr Func AddC<int8_t>::id() { return Func::AddC_8s; }
    template <> constexpr Func AddC<uint8_t>::id() { return Func::AddC_8u; }
    template <> constexpr Func AddC<int16_t>::id() { return Func::AddC_16s; }
    template <> constexpr Func AddC<uint16_t>::id() { return Func::AddC_16u; }
    template <> constexpr Func AddC<int32_t>::id() { return Func::AddC_32s; }
    template <> constexpr Func AddC<uint32_t>::id() { return Func::AddC_32u; }
    template <> constexpr Func AddC<int64_t>::id() { return Func::AddC_64s; }
    template <> constexpr Func AddC<uint64_t>::id() { return Func::AddC_64u; }
    template <> constexpr Func AddC<float>::id() { return Func::AddC_32f; }
    template <> constexpr Func AddC<double>::id() { return Func::AddC_64f; }
    template <> constexpr const char * AddC<int8_t>::programName() { return "addC_8s"; }
    template <> constexpr const char * AddC<uint8_t>::programName() { return "addC_8u"; }
    template <> constexpr const char * AddC<int16_t>::programName() { return "addC_16s"; }
    template <> constexpr const char * AddC<uint16_t>::programName() { return "addC_16u"; }
    template <> constexpr const char * AddC<int32_t>::programName() { return "addC_32s"; }
    template <> constexpr const char * AddC<uint32_t>::programName() { return "addC_32u"; }
    template <> constexpr const char * AddC<int64_t>::programName() { return "addC_64s"; }
    template <> constexpr const char * AddC<uint64_t>::programName() { return "addC_64u"; }
    template <> constexpr const char * AddC<float>::programName() { return "addC_32f"; }
    template <> constexpr const char * AddC<double>::programName() { return "addC_64f"; }

    template <typename _T>
    struct SubC : public PtrValPtr<_T>
    {
        static constexpr Func id();
        static constexpr const char * programName();
        static const char * programText()
        {
            static std::string text = TextProgram::func3args<const _T*, _T, _T*>(programName(), PtrValPtr<_T>::opSubC());
            return text.data();
        }
    };

    template <> constexpr Func SubC<int8_t>::id() { return Func::SubC_8s; }
    template <> constexpr Func SubC<uint8_t>::id() { return Func::SubC_8u; }
    template <> constexpr Func SubC<int16_t>::id() { return Func::SubC_16s; }
    template <> constexpr Func SubC<uint16_t>::id() { return Func::SubC_16u; }
    template <> constexpr Func SubC<int32_t>::id() { return Func::SubC_32s; }
    template <> constexpr Func SubC<uint32_t>::id() { return Func::SubC_32u; }
    template <> constexpr Func SubC<int64_t>::id() { return Func::SubC_64s; }
    template <> constexpr Func SubC<uint64_t>::id() { return Func::SubC_64u; }
    template <> constexpr Func SubC<float>::id() { return Func::SubC_32f; }
    template <> constexpr Func SubC<double>::id() { return Func::SubC_64f; }
    template <> constexpr const char * SubC<int8_t>::programName() { return "subC_8s"; }
    template <> constexpr const char * SubC<uint8_t>::programName() { return "subC_8u"; }
    template <> constexpr const char * SubC<int16_t>::programName() { return "subC_16s"; }
    template <> constexpr const char * SubC<uint16_t>::programName() { return "subC_16u"; }
    template <> constexpr const char * SubC<int32_t>::programName() { return "subC_32s"; }
    template <> constexpr const char * SubC<uint32_t>::programName() { return "subC_32u"; }
    template <> constexpr const char * SubC<int64_t>::programName() { return "subC_64s"; }
    template <> constexpr const char * SubC<uint64_t>::programName() { return "subC_64u"; }
    template <> constexpr const char * SubC<float>::programName() { return "subC_32f"; }
    template <> constexpr const char * SubC<double>::programName() { return "subC_64f"; }

    template <typename _T>
    struct SubCRev : public PtrValPtr<_T>
    {
        static constexpr Func id();
        static constexpr const char * programName();
        static const char * programText()
        {
            static std::string text = TextProgram::func3args<const _T*, _T, _T*>(programName(), PtrValPtr<_T>::opSubCRev());
            return text.data();
        }
    };

    template <> constexpr Func SubCRev<int8_t>::id() { return Func::SubCRev_8s; }
    template <> constexpr Func SubCRev<uint8_t>::id() { return Func::SubCRev_8u; }
    template <> constexpr Func SubCRev<int16_t>::id() { return Func::SubCRev_16s; }
    template <> constexpr Func SubCRev<uint16_t>::id() { return Func::SubCRev_16u; }
    template <> constexpr Func SubCRev<int32_t>::id() { return Func::SubCRev_32s; }
    template <> constexpr Func SubCRev<uint32_t>::id() { return Func::SubCRev_32u; }
    template <> constexpr Func SubCRev<int64_t>::id() { return Func::SubCRev_64s; }
    template <> constexpr Func SubCRev<uint64_t>::id() { return Func::SubCRev_64u; }
    template <> constexpr Func SubCRev<float>::id() { return Func::SubCRev_32f; }
    template <> constexpr Func SubCRev<double>::id() { return Func::SubCRev_64f; }
    template <> constexpr const char * SubCRev<int8_t>::programName() { return "subCRev_8s"; }
    template <> constexpr const char * SubCRev<uint8_t>::programName() { return "subCRev_8u"; }
    template <> constexpr const char * SubCRev<int16_t>::programName() { return "subCRev_16s"; }
    template <> constexpr const char * SubCRev<uint16_t>::programName() { return "subCRev_16u"; }
    template <> constexpr const char * SubCRev<int32_t>::programName() { return "subCRev_32s"; }
    template <> constexpr const char * SubCRev<uint32_t>::programName() { return "subCRev_32u"; }
    template <> constexpr const char * SubCRev<int64_t>::programName() { return "subCRev_64s"; }
    template <> constexpr const char * SubCRev<uint64_t>::programName() { return "subCRev_64u"; }
    template <> constexpr const char * SubCRev<float>::programName() { return "subCRev_32f"; }
    template <> constexpr const char * SubCRev<double>::programName() { return "subCRev_64f"; }

    template <typename _T>
    struct MulC : public PtrValPtr<_T>
    {
        static constexpr Func id();
        static constexpr const char * programName();
        static const char * programText()
        {
            static std::string text = TextProgram::func3args<const _T*, _T, _T*>(programName(), PtrValPtr<_T>::opMulC());
            return text.data();
        }
    };

    template <> constexpr Func MulC<int8_t>::id() { return Func::MulC_8s; }
    template <> constexpr Func MulC<uint8_t>::id() { return Func::MulC_8u; }
    template <> constexpr Func MulC<int16_t>::id() { return Func::MulC_16s; }
    template <> constexpr Func MulC<uint16_t>::id() { return Func::MulC_16u; }
    template <> constexpr Func MulC<int32_t>::id() { return Func::MulC_32s; }
    template <> constexpr Func MulC<uint32_t>::id() { return Func::MulC_32u; }
    template <> constexpr Func MulC<int64_t>::id() { return Func::MulC_64s; }
    template <> constexpr Func MulC<uint64_t>::id() { return Func::MulC_64u; }
    template <> constexpr Func MulC<float>::id() { return Func::MulC_32f; }
    template <> constexpr Func MulC<double>::id() { return Func::MulC_64f; }
    template <> constexpr const char * MulC<int8_t>::programName() { return "mulC_8s"; }
    template <> constexpr const char * MulC<uint8_t>::programName() { return "mulC_8u"; }
    template <> constexpr const char * MulC<int16_t>::programName() { return "mulC_16s"; }
    template <> constexpr const char * MulC<uint16_t>::programName() { return "mulC_16u"; }
    template <> constexpr const char * MulC<int32_t>::programName() { return "mulC_32s"; }
    template <> constexpr const char * MulC<uint32_t>::programName() { return "mulC_32u"; }
    template <> constexpr const char * MulC<int64_t>::programName() { return "mulC_64s"; }
    template <> constexpr const char * MulC<uint64_t>::programName() { return "mulC_64u"; }
    template <> constexpr const char * MulC<float>::programName() { return "mulC_32f"; }
    template <> constexpr const char * MulC<double>::programName() { return "mulC_64f"; }

    template <typename _T>
    struct DivC : public PtrValPtr<_T>
    {
        static constexpr Func id();
        static constexpr const char * programName();
        static const char * programText()
        {
            static std::string text = TextProgram::func3args<const _T*, _T, _T*>(programName(), PtrValPtr<_T>::opDivC());
            return text.data();
        }
    };

    template <> constexpr Func DivC<int8_t>::id() { return Func::DivC_8s; }
    template <> constexpr Func DivC<uint8_t>::id() { return Func::DivC_8u; }
    template <> constexpr Func DivC<int16_t>::id() { return Func::DivC_16s; }
    template <> constexpr Func DivC<uint16_t>::id() { return Func::DivC_16u; }
    template <> constexpr Func DivC<int32_t>::id() { return Func::DivC_32s; }
    template <> constexpr Func DivC<uint32_t>::id() { return Func::DivC_32u; }
    template <> constexpr Func DivC<int64_t>::id() { return Func::DivC_64s; }
    template <> constexpr Func DivC<uint64_t>::id() { return Func::DivC_64u; }
    template <> constexpr Func DivC<float>::id() { return Func::DivC_32f; }
    template <> constexpr Func DivC<double>::id() { return Func::DivC_64f; }
    template <> constexpr const char * DivC<int8_t>::programName() { return "divC_8s"; }
    template <> constexpr const char * DivC<uint8_t>::programName() { return "divC_8u"; }
    template <> constexpr const char * DivC<int16_t>::programName() { return "divC_16s"; }
    template <> constexpr const char * DivC<uint16_t>::programName() { return "divC_16u"; }
    template <> constexpr const char * DivC<int32_t>::programName() { return "divC_32s"; }
    template <> constexpr const char * DivC<uint32_t>::programName() { return "divC_32u"; }
    template <> constexpr const char * DivC<int64_t>::programName() { return "divC_64s"; }
    template <> constexpr const char * DivC<uint64_t>::programName() { return "divC_64u"; }
    template <> constexpr const char * DivC<float>::programName() { return "divC_32f"; }
    template <> constexpr const char * DivC<double>::programName() { return "divC_64f"; }

    template <typename _T>
    struct DivCRev : public PtrValPtr<_T>
    {
        static constexpr Func id();
        static constexpr const char * programName();
        static const char * programText()
        {
            static std::string text = TextProgram::func3args<const _T*, _T, _T*>(programName(), PtrValPtr<_T>::opDivCRev());
            return text.data();
        }
    };

    template <> constexpr Func DivCRev<int8_t>::id() { return Func::DivCRev_8s; }
    template <> constexpr Func DivCRev<uint8_t>::id() { return Func::DivCRev_8u; }
    template <> constexpr Func DivCRev<int16_t>::id() { return Func::DivCRev_16s; }
    template <> constexpr Func DivCRev<uint16_t>::id() { return Func::DivCRev_16u; }
    template <> constexpr Func DivCRev<int32_t>::id() { return Func::DivCRev_32s; }
    template <> constexpr Func DivCRev<uint32_t>::id() { return Func::DivCRev_32u; }
    template <> constexpr Func DivCRev<int64_t>::id() { return Func::DivCRev_64s; }
    template <> constexpr Func DivCRev<uint64_t>::id() { return Func::DivCRev_64u; }
    template <> constexpr Func DivCRev<float>::id() { return Func::DivCRev_32f; }
    template <> constexpr Func DivCRev<double>::id() { return Func::DivCRev_64f; }
    template <> constexpr const char * DivCRev<int8_t>::programName() { return "divCRev_8s"; }
    template <> constexpr const char * DivCRev<uint8_t>::programName() { return "divCRev_8u"; }
    template <> constexpr const char * DivCRev<int16_t>::programName() { return "divCRev_16s"; }
    template <> constexpr const char * DivCRev<uint16_t>::programName() { return "divCRev_16u"; }
    template <> constexpr const char * DivCRev<int32_t>::programName() { return "divCRev_32s"; }
    template <> constexpr const char * DivCRev<uint32_t>::programName() { return "divCRev_32u"; }
    template <> constexpr const char * DivCRev<int64_t>::programName() { return "divCRev_64s"; }
    template <> constexpr const char * DivCRev<uint64_t>::programName() { return "divCRev_64u"; }
    template <> constexpr const char * DivCRev<float>::programName() { return "divCRev_32f"; }
    template <> constexpr const char * DivCRev<double>::programName() { return "divCRev_64f"; }

    template <typename _T>
    struct Abs : public PtrValPtr<_T>
    {
        static constexpr Func id();
        static constexpr const char * programName();
        static const char * programText()
        {
            static std::string text = TextProgram::func3args<const _T*, _T, _T*>(programName(), PtrValPtr<_T>::opAbs());
            return text.data();
        }
    };

    template <> constexpr Func Abs<int8_t>::id() { return Func::Abs_8s; }
    template <> constexpr Func Abs<int16_t>::id() { return Func::Abs_16s; }
    template <> constexpr Func Abs<int32_t>::id() { return Func::Abs_32s; }
    template <> constexpr Func Abs<int64_t>::id() { return Func::Abs_64s; }
    template <> constexpr Func Abs<float>::id() { return Func::Abs_32f; }
    template <> constexpr Func Abs<double>::id() { return Func::Abs_64f; }
    template <> constexpr const char * Abs<int8_t>::programName() { return "abs_8s"; }
    template <> constexpr const char * Abs<int16_t>::programName() { return "abs_16s"; }
    template <> constexpr const char * Abs<int32_t>::programName() { return "abs_32s"; }
    template <> constexpr const char * Abs<int64_t>::programName() { return "abs_64s"; }
    template <> constexpr const char * Abs<float>::programName() { return "abs_32f"; }
    template <> constexpr const char * Abs<double>::programName() { return "abs_64f"; }

    //

    inline constexpr TextProgram program(Func func)
    {
        switch (func) {
            case Func::Add_8s: return TextProgram::create<Add<int8_t>>();
            case Func::Sub_8s: return TextProgram::create<Sub<int8_t>>();
            case Func::Mul_8s: return TextProgram::create<Mul<int8_t>>();
            case Func::Div_8s: return TextProgram::create<Div<int8_t>>();
            case Func::AddC_8s: return TextProgram::create<AddC<int8_t>>();
            case Func::SubC_8s: return TextProgram::create<SubC<int8_t>>();
            case Func::MulC_8s: return TextProgram::create<MulC<int8_t>>();
            case Func::DivC_8s: return TextProgram::create<DivC<int8_t>>();
            case Func::SubCRev_8s: return TextProgram::create<SubCRev<int8_t>>();
            case Func::DivCRev_8s: return TextProgram::create<DivCRev<int8_t>>();
            //
            case Func::Add_8u: return TextProgram::create<Add<uint8_t>>();
            case Func::Sub_8u: return TextProgram::create<Sub<uint8_t>>();
            case Func::Mul_8u: return TextProgram::create<Mul<uint8_t>>();
            case Func::Div_8u: return TextProgram::create<Div<uint8_t>>();
            case Func::AddC_8u: return TextProgram::create<AddC<uint8_t>>();
            case Func::SubC_8u: return TextProgram::create<SubC<uint8_t>>();
            case Func::MulC_8u: return TextProgram::create<MulC<uint8_t>>();
            case Func::DivC_8u: return TextProgram::create<DivC<uint8_t>>();
            case Func::SubCRev_8u: return TextProgram::create<SubCRev<uint8_t>>();
            case Func::DivCRev_8u: return TextProgram::create<DivCRev<uint8_t>>();
            //
            case Func::Add_16s: return TextProgram::create<Add<int16_t>>();
            case Func::Sub_16s: return TextProgram::create<Sub<int16_t>>();
            case Func::Mul_16s: return TextProgram::create<Mul<int16_t>>();
            case Func::Div_16s: return TextProgram::create<Div<int16_t>>();
            case Func::AddC_16s: return TextProgram::create<AddC<int16_t>>();
            case Func::SubC_16s: return TextProgram::create<SubC<int16_t>>();
            case Func::MulC_16s: return TextProgram::create<MulC<int16_t>>();
            case Func::DivC_16s: return TextProgram::create<DivC<int16_t>>();
            case Func::SubCRev_16s: return TextProgram::create<SubCRev<int16_t>>();
            case Func::DivCRev_16s: return TextProgram::create<DivCRev<int16_t>>();
            //
            case Func::Add_16u: return TextProgram::create<Add<uint16_t>>();
            case Func::Sub_16u: return TextProgram::create<Sub<uint16_t>>();
            case Func::Mul_16u: return TextProgram::create<Mul<uint16_t>>();
            case Func::Div_16u: return TextProgram::create<Div<uint16_t>>();
            case Func::AddC_16u: return TextProgram::create<AddC<uint16_t>>();
            case Func::SubC_16u: return TextProgram::create<SubC<uint16_t>>();
            case Func::MulC_16u: return TextProgram::create<MulC<uint16_t>>();
            case Func::DivC_16u: return TextProgram::create<DivC<uint16_t>>();
            case Func::SubCRev_16u: return TextProgram::create<SubCRev<uint16_t>>();
            case Func::DivCRev_16u: return TextProgram::create<DivCRev<uint16_t>>();
            //
            case Func::Add_32s: return TextProgram::create<Add<int32_t>>();
            case Func::Sub_32s: return TextProgram::create<Sub<int32_t>>();
            case Func::Mul_32s: return TextProgram::create<Mul<int32_t>>();
            case Func::Div_32s: return TextProgram::create<Div<int32_t>>();
            case Func::AddC_32s: return TextProgram::create<AddC<int32_t>>();
            case Func::SubC_32s: return TextProgram::create<SubC<int32_t>>();
            case Func::MulC_32s: return TextProgram::create<MulC<int32_t>>();
            case Func::DivC_32s: return TextProgram::create<DivC<int32_t>>();
            case Func::SubCRev_32s: return TextProgram::create<SubCRev<int32_t>>();
            case Func::DivCRev_32s: return TextProgram::create<DivCRev<int32_t>>();
            //
            case Func::Add_32u: return TextProgram::create<Add<uint32_t>>();
            case Func::Sub_32u: return TextProgram::create<Sub<uint32_t>>();
            case Func::Mul_32u: return TextProgram::create<Mul<uint32_t>>();
            case Func::Div_32u: return TextProgram::create<Div<uint32_t>>();
            case Func::AddC_32u: return TextProgram::create<AddC<uint32_t>>();
            case Func::SubC_32u: return TextProgram::create<SubC<uint32_t>>();
            case Func::MulC_32u: return TextProgram::create<MulC<uint32_t>>();
            case Func::DivC_32u: return TextProgram::create<DivC<uint32_t>>();
            case Func::SubCRev_32u: return TextProgram::create<SubCRev<uint32_t>>();
            case Func::DivCRev_32u: return TextProgram::create<DivCRev<uint32_t>>();
            //
            case Func::Add_64s: return TextProgram::create<Add<int64_t>>();
            case Func::Sub_64s: return TextProgram::create<Sub<int64_t>>();
            case Func::Mul_64s: return TextProgram::create<Mul<int64_t>>();
            case Func::Div_64s: return TextProgram::create<Div<int64_t>>();
            case Func::AddC_64s: return TextProgram::create<AddC<int64_t>>();
            case Func::SubC_64s: return TextProgram::create<SubC<int64_t>>();
            case Func::MulC_64s: return TextProgram::create<MulC<int64_t>>();
            case Func::DivC_64s: return TextProgram::create<DivC<int64_t>>();
            case Func::SubCRev_64s: return TextProgram::create<SubCRev<int64_t>>();
            case Func::DivCRev_64s: return TextProgram::create<DivCRev<int64_t>>();
            //
            case Func::Add_64u: return TextProgram::create<Add<uint64_t>>();
            case Func::Sub_64u: return TextProgram::create<Sub<uint64_t>>();
            case Func::Mul_64u: return TextProgram::create<Mul<uint64_t>>();
            case Func::Div_64u: return TextProgram::create<Div<uint64_t>>();
            case Func::AddC_64u: return TextProgram::create<AddC<uint64_t>>();
            case Func::SubC_64u: return TextProgram::create<SubC<uint64_t>>();
            case Func::MulC_64u: return TextProgram::create<MulC<uint64_t>>();
            case Func::DivC_64u: return TextProgram::create<DivC<uint64_t>>();
            case Func::SubCRev_64u: return TextProgram::create<SubCRev<uint64_t>>();
            case Func::DivCRev_64u: return TextProgram::create<DivCRev<uint64_t>>();
            //
            case Func::Add_32f: return TextProgram::create<Add<float>>();
            case Func::Sub_32f: return TextProgram::create<Sub<float>>();
            case Func::Mul_32f: return TextProgram::create<Mul<float>>();
            case Func::Div_32f: return TextProgram::create<Div<float>>();
            case Func::AddC_32f: return TextProgram::create<AddC<float>>();
            case Func::SubC_32f: return TextProgram::create<SubC<float>>();
            case Func::MulC_32f: return TextProgram::create<MulC<float>>();
            case Func::DivC_32f: return TextProgram::create<DivC<float>>();
            case Func::SubCRev_32f: return TextProgram::create<SubCRev<float>>();
            case Func::DivCRev_32f: return TextProgram::create<DivCRev<float>>();
            //
            case Func::Add_64f: return TextProgram::create<Add<double>>();
            case Func::Sub_64f: return TextProgram::create<Sub<double>>();
            case Func::Mul_64f: return TextProgram::create<Mul<double>>();
            case Func::Div_64f: return TextProgram::create<Div<double>>();
            case Func::AddC_64f: return TextProgram::create<AddC<double>>();
            case Func::SubC_64f: return TextProgram::create<SubC<double>>();
            case Func::MulC_64f: return TextProgram::create<MulC<double>>();
            case Func::DivC_64f: return TextProgram::create<DivC<double>>();
            case Func::SubCRev_64f: return TextProgram::create<SubCRev<double>>();
            case Func::DivCRev_64f: return TextProgram::create<DivCRev<double>>();
            //
            case Func::Abs_8s: return TextProgram::create<Abs<int8_t>>();
            case Func::Abs_16s: return TextProgram::create<Abs<int16_t>>();
            case Func::Abs_32s: return TextProgram::create<Abs<int32_t>>();
            case Func::Abs_64s: return TextProgram::create<Abs<int64_t>>();
            case Func::Abs_32f: return TextProgram::create<Abs<float>>();
            case Func::Abs_64f: return TextProgram::create<Abs<double>>();
            case Func::Count:
                break;
        }
        return {nullptr, nullptr};
    }
}
} // internals
} // ocl
