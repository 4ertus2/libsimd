#include "ocl.h"

#include <vector>
#include <cstring>
#include <memory>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

namespace ocl
{
namespace internals
{
    ///
    template <typename _T>
    class SrcDstBuffers
    {
    public:
        using DataType = _T;
        using ClMemType = std::remove_reference<decltype(*cl_mem())>::type; // _cl_mem

        SrcDstBuffers(cl_context gpuContext, uint32_t workSize)
        :   workSize_(workSize)
        {
            cl_int err = 0;
            srcA_ = std::shared_ptr<ClMemType>(
                clCreateBuffer(gpuContext, CL_MEM_READ_ONLY, dataSize(), nullptr, &err), clReleaseMemObject);
            if (err)
                throw OCL_EXCEPTION;

            dst_ = std::shared_ptr<ClMemType>(
                clCreateBuffer(gpuContext, CL_MEM_WRITE_ONLY, dataSize(), nullptr, &err), clReleaseMemObject);
            if (err)
                throw OCL_EXCEPTION;
        }

        uint32_t workSize() const { return workSize_; }
        size_t dataSize() const { return workSize_ * sizeof(_T); }

        cl_mem srcA() { return srcA_.get(); }
        cl_mem dst() { return dst_.get(); }

    private:
        uint32_t workSize_;
        std::shared_ptr<ClMemType> srcA_;
        std::shared_ptr<ClMemType> dst_;
    };

    ///
    template <typename _T>
    class SrcSrcDstBuffers : public SrcDstBuffers<_T>
    {
    public:
        using ClMemType = typename SrcDstBuffers<_T>::ClMemType;

        SrcSrcDstBuffers(cl_context gpuContext, uint32_t workSize)
        :   SrcDstBuffers<_T>(gpuContext, workSize)
        {
            cl_int err = 0;
            srcB_ = std::shared_ptr<ClMemType>(
                clCreateBuffer(gpuContext, CL_MEM_READ_ONLY, SrcDstBuffers<_T>::dataSize(), nullptr, &err),
                clReleaseMemObject);
            if (err)
                throw OCL_EXCEPTION;
        }

        cl_mem srcB() { return srcB_.get(); }

    private:
        std::shared_ptr<ClMemType> srcB_;
    };

    ///
    template <typename _T>
    class SrcValDstBuffers : public SrcDstBuffers<_T>
    {
    public:
        using ClMemType = typename SrcDstBuffers<_T>::ClMemType;

        SrcValDstBuffers(cl_context gpuContext, uint32_t workSize)
        :   SrcDstBuffers<_T>(gpuContext, workSize)
        {}
    };

    ///
    struct Kernel
    {
        enum class Func
        {
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
            Count
        };

        static constexpr uint32_t funcCount() { return (uint32_t)Func::Count; }

        struct FloatPPP
        {
            using DataTypeSrc1 = const float*;
            using DataTypeSrc2 = const float*;
            using DataTypeDst = float*;
            using BufferType = SrcSrcDstBuffers<float>;
        };

        struct FloatPVP
        {
            using DataTypeSrc1 = const float*;
            using DataTypeSrc2 = float;
            using DataTypeDst = float*;
            using BufferType = SrcValDstBuffers<float>;
        };

        struct Add_32f : public FloatPPP
        {
            static constexpr const Func id() { return Func::Add_32f; }
            static constexpr const char * programName() { return "add_32f"; }
            static constexpr const char * programText()
            {
                return R"(
                __kernel void add_32f(__global const float* a, __global const float* b, __global float* c, int size)
                {
                    int iGID = get_global_id(0);
                    if (iGID >= size)
                        return;
                    c[iGID] = a[iGID] + b[iGID];
                })";
            }
        };

        struct Sub_32f : public FloatPPP
        {
            static constexpr const Func id() { return Func::Sub_32f; }
            static constexpr const char * programName() { return "sub_32f"; }
            static constexpr const char * programText()
            {
                return R"(
                __kernel void sub_32f(__global const float* a, __global const float* b, __global float* c, int size)
                {
                    int iGID = get_global_id(0);
                    if (iGID >= size)
                        return;
                    c[iGID] = a[iGID] - b[iGID];
                })";
            }
        };

        struct Mul_32f : public FloatPPP
        {
            static constexpr const Func id() { return Func::Mul_32f; }
            static constexpr const char * programName() { return "mul_32f"; }
            static constexpr const char * programText()
            {
                return R"(
                __kernel void mul_32f(__global const float* a, __global const float* b, __global float* c, int size)
                {
                    int iGID = get_global_id(0);
                    if (iGID >= size)
                        return;
                    c[iGID] = a[iGID] * b[iGID];
                })";
            }
        };

        struct Div_32f : public FloatPPP
        {
            static constexpr const Func id() { return Func::Div_32f; }
            static constexpr const char * programName() { return "div_32f"; }
            static constexpr const char * programText()
            {
                return R"(
                __kernel void div_32f(__global const float* a, __global const float* b, __global float* c, int size)
                {
                    int iGID = get_global_id(0);
                    if (iGID >= size)
                        return;
                    c[iGID] = a[iGID] / b[iGID];
                })";
            }
        };

        struct AddC_32f : public FloatPVP
        {
            static constexpr const Func id() { return Func::AddC_32f; }
            static constexpr const char * programName() { return "addC_32f"; }
            static constexpr const char * programText()
            {
                return R"(
                __kernel void addC_32f(__global const float* a, float b, __global float* c, int size)
                {
                    int iGID = get_global_id(0);
                    if (iGID >= size)
                        return;
                    c[iGID] = a[iGID] + b;
                })";
            }
        };

        struct SubC_32f : public FloatPVP
        {
            static constexpr const Func id() { return Func::SubC_32f; }
            static constexpr const char * programName() { return "subC_32f"; }
            static constexpr const char * programText()
            {
                return R"(
                __kernel void subC_32f(__global const float* a, float b, __global float* c, int size)
                {
                    int iGID = get_global_id(0);
                    if (iGID >= size)
                        return;
                    c[iGID] = a[iGID] - b;
                })";
            }
        };

        struct SubCRev_32f : public FloatPVP
        {
            static constexpr const Func id() { return Func::SubCRev_32f; }
            static constexpr const char * programName() { return "subCRev_32f"; }
            static constexpr const char * programText()
            {
                return R"(
                __kernel void subCRev_32f(__global const float* a, float b, __global float* c, int size)
                {
                    int iGID = get_global_id(0);
                    if (iGID >= size)
                        return;
                    c[iGID] = b - a[iGID];
                })";
            }
        };

        struct MulC_32f : public FloatPVP
        {
            static constexpr const Func id() { return Func::MulC_32f; }
            static constexpr const char * programName() { return "mulC_32f"; }
            static constexpr const char * programText()
            {
                return R"(
                __kernel void mulC_32f(__global const float* a, float b, __global float* c, int size)
                {
                    int iGID = get_global_id(0);
                    if (iGID >= size)
                        return;
                    c[iGID] = a[iGID] * b;
                })";
            }
        };

        struct DivC_32f : public FloatPVP
        {
            static constexpr const Func id() { return Func::DivC_32f; }
            static constexpr const char * programName() { return "divC_32f"; }
            static constexpr const char * programText()
            {
                return R"(
                __kernel void divC_32f(__global const float* a, float b, __global float* c, int size)
                {
                    int iGID = get_global_id(0);
                    if (iGID >= size)
                        return;
                    c[iGID] = a[iGID] / b;
                })";
            }
        };

        struct DivCRev_32f : public FloatPVP
        {
            static constexpr const Func id() { return Func::DivCRev_32f; }
            static constexpr const char * programName() { return "divCRev_32f"; }
            static constexpr const char * programText()
            {
                return R"(
                __kernel void divCRev_32f(__global const float* a, float b, __global float* c, int size)
                {
                    int iGID = get_global_id(0);
                    if (iGID >= size)
                        return;
                    c[iGID] = b / a[iGID];
                })";
            }
        };

        struct Abs_32f : public FloatPVP
        {
            static constexpr const Func id() { return Func::Abs_32f; }
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

        static constexpr const char * programName(Func func)
        {
            switch (func) {
                case Func::Add_32f: return Add_32f::programName();
                case Func::Sub_32f: return Sub_32f::programName();
                case Func::Mul_32f: return Mul_32f::programName();
                case Func::Div_32f: return Div_32f::programName();
                //
                case Func::AddC_32f: return AddC_32f::programName();
                case Func::SubC_32f: return SubC_32f::programName();
                case Func::MulC_32f: return MulC_32f::programName();
                case Func::DivC_32f: return DivC_32f::programName();
                case Func::SubCRev_32f: return SubCRev_32f::programName();
                case Func::DivCRev_32f: return DivCRev_32f::programName();
                //
                case Func::Abs_32f: return Abs_32f::programName();
                case Func::Count:
                    break;
            }
            return nullptr;
        }

        static constexpr const char * programText(Func func)
        {
            switch (func) {
                case Func::Add_32f: return Add_32f::programText();
                case Func::Sub_32f: return Sub_32f::programText();
                case Func::Mul_32f: return Mul_32f::programText();
                case Func::Div_32f: return Div_32f::programText();
                //
                case Func::AddC_32f: return AddC_32f::programText();
                case Func::SubC_32f: return SubC_32f::programText();
                case Func::MulC_32f: return MulC_32f::programText();
                case Func::DivC_32f: return DivC_32f::programText();
                case Func::SubCRev_32f: return SubCRev_32f::programText();
                case Func::DivCRev_32f: return DivCRev_32f::programText();
                //
                case Func::Abs_32f: return Abs_32f::programText();
                case Func::Count:
                    break;
            }
            return nullptr;
        }
    };

    ///
    class SimdOpenCl
    {
    public:
        static SimdOpenCl& getInstance()
        {
            static uint32_t platformId = 1;
            static uint32_t deviceId = 1;
            static SimdOpenCl instance(platformId, deviceId);
            return instance;
        }

        static size_t alignedSize(size_t requiredSize)
        {
            if (requiredSize == 0)
                return localWorkSize();
#if 1
            return requiredSize + (~requiredSize + 1) % localWorkSize();
#else
            size_t r = requiredSize % localWorkSize();
            return requiredSize + (r ? (localWorkSize() - r) : 0);
#endif
        }

        template <typename _T>
        void exec(typename _T::DataTypeSrc1 src1, typename _T::DataTypeSrc2 src2,
                typename _T::DataTypeDst dst, int size)
        {
            uint32_t workSize = alignedSize(size);
            typename _T::BufferType buffers(gpuContext(), workSize);
            //size_t dataSize = buffers.dataSize();

            cl_kernel kernel = programs_[(uint32_t)_T::id()].kernel();
            setArgs(kernel, buffers, size);

            runCoreSequence(src1, src2, dst, kernel, buffers);
        }

    private:
        using ContextType = std::remove_reference<decltype(*cl_context())>::type;           // _cl_context
        using CmdQueueType = std::remove_reference<decltype(*cl_command_queue())>::type;    // _cl_command_queue
        using ProgramType = std::remove_reference<decltype(*cl_program())>::type;           // _cl_program
        using KernelType = std::remove_reference<decltype(*cl_kernel())>::type;             // _cl_kernel

        ///
        class BuiltProgram
        {
        public:
            void setProgram(cl_program prog) { program_ = std::shared_ptr<ProgramType>(prog, clReleaseProgram); }
            void setKernel(cl_kernel kern) { kernel_ = std::shared_ptr<KernelType>(kern, clReleaseKernel); }

            cl_program program() { return program_.get(); }
            cl_kernel kernel() { return kernel_.get(); }

        private:
            std::shared_ptr<ProgramType> program_;
            std::shared_ptr<KernelType> kernel_;
        };

        cl_platform_id platform_ = nullptr;
        cl_device_id device_ = nullptr;
        std::shared_ptr<ContextType> gpuContext_;
        std::shared_ptr<CmdQueueType> commandQueue_;
        std::vector<BuiltProgram> programs_; // implicit hash_map<(uint32_t)Kernel::Func, BuiltProgram>

        static constexpr uint32_t localWorkSize() { return 256; }
        cl_context gpuContext() { return gpuContext_.get(); }
        cl_command_queue commandQueue() { return commandQueue_.get(); }

        SimdOpenCl(uint32_t platformId, uint32_t deviceId)
        {
            cl_uint numPlatforms;
            if (clGetPlatformIDs(0, nullptr, &numPlatforms) || numPlatforms < platformId)
                throw OCL_EXCEPTION;
            if (clGetPlatformIDs(platformId, &platform_, nullptr))
                throw OCL_EXCEPTION;

            cl_uint numDevices;
            if (clGetDeviceIDs(platform_, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices) || numDevices < deviceId)
                throw OCL_EXCEPTION;
            if (clGetDeviceIDs(platform_, CL_DEVICE_TYPE_GPU, deviceId, &device_, nullptr))
                throw OCL_EXCEPTION;

            cl_int err = 0;
            gpuContext_ = std::shared_ptr<ContextType>(
                clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err), clReleaseContext);
            if (err)
                throw OCL_EXCEPTION;

#ifdef CL_USE_DEPRECATED_OPENCL_1_2_APIS
            commandQueue_ = std::shared_ptr<CmdQueueType>(
                clCreateCommandQueue(gpuContext(), device_, 0, &err), clReleaseCommandQueue);
#else
            commandQueue_ = std::shared_ptr<CmdQueueType>(
                clCreateCommandQueueWithProperties(gpuContext(), device_, nullptr, &err), clReleaseCommandQueue);
#endif
            if (err)
                throw OCL_EXCEPTION;

            makePrograms();
        }

        void makePrograms()
        {
            programs_.resize(Kernel::funcCount());
            for (uint32_t i = 0; i < Kernel::funcCount(); ++i) {
                makeKernel((Kernel::Func)i, programs_[i]);
            }
        }

        void makeKernel(Kernel::Func func, BuiltProgram& prog)
        {
            cl_int err = 0;
            const char * text = Kernel::programText(func);
            size_t progLength = strlen(text);

            if (programs_.size() <= (uint32_t)func)
                throw OCL_EXCEPTION;

            prog.setProgram(clCreateProgramWithSource(gpuContext(), 1, (const char **)&text, &progLength, &err));
            if (err)
                throw OCL_EXCEPTION;

            //const char* flags = "-cl-fast-relaxed-math";
            err = clBuildProgram(prog.program(), 0, nullptr, nullptr, nullptr, nullptr);
            if (err)
                throw OCL_EXCEPTION;

            prog.setKernel(clCreateKernel(prog.program(), Kernel::programName(func), &err));
            if (err)
                throw OCL_EXCEPTION;
        }

        template <typename _T>
        void setArgs(cl_kernel& kernel, SrcSrcDstBuffers<_T>& bufs, uint32_t count)
        {
            cl_mem srcA = bufs.srcA();
            cl_mem srcB = bufs.srcB();
            cl_mem dst = bufs.dst();

            cl_int err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &srcA);
            err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &srcB);
            err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &dst);
            err |= clSetKernelArg(kernel, 3, sizeof(cl_int), &count);
            if (err)
                throw OCL_EXCEPTION;
        }

        template <typename _T>
        void setArgs(cl_kernel& kernel, SrcValDstBuffers<_T>& bufs, uint32_t count)
        {
            cl_mem srcA = bufs.srcA();
            cl_mem dst = bufs.dst();

            cl_int err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &srcA);
            //err |= clSetKernelArg(kernel, 1, sizeof(_T), &bufs.srcB);
            err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &dst);
            err |= clSetKernelArg(kernel, 3, sizeof(cl_int), &count);
            if (err)
                throw OCL_EXCEPTION;
        }

        template <typename _T>
        void runCoreSequence(const _T * src1, const _T * src2, _T * dst, cl_kernel kernel, SrcSrcDstBuffers<_T>& bufs)
        {
            asyncWriteToGPU(src1, bufs.dataSize(), bufs.srcA());
            asyncWriteToGPU(src2, bufs.dataSize(), bufs.srcB());
            launchKernel(kernel, bufs.workSize());
            syncReadFromGPU(dst, bufs.dataSize(), bufs.dst());
        }

        template <typename _T>
        void runCoreSequence(const _T * src1, _T src2, _T * dst, cl_kernel kernel, SrcValDstBuffers<_T>& bufs)
        {
            cl_int err = clSetKernelArg(kernel, 1, sizeof(_T), &src2);
            if (err)
                throw OCL_EXCEPTION;

            asyncWriteToGPU(src1, bufs.dataSize(), bufs.srcA());
            launchKernel(kernel, bufs.workSize());
            syncReadFromGPU(dst, bufs.dataSize(), bufs.dst());
        }

        void launchKernel(cl_kernel kernel, size_t workSize)
        {
            size_t localWS = localWorkSize();
            cl_int err = clEnqueueNDRangeKernel(
                commandQueue(), kernel, 1, nullptr, &workSize, &localWS, 0, nullptr, nullptr);
            if (err)
                throw OCL_EXCEPTION;
        }

        void asyncWriteToGPU(const void * src, size_t dataSize, cl_mem gpuBuffer)
        {
            cl_int err = clEnqueueWriteBuffer(
                commandQueue(), gpuBuffer, CL_FALSE, 0, dataSize, src, 0, nullptr, nullptr);
            if (err)
                throw OCL_EXCEPTION;
        }

        void syncReadFromGPU(void * dst, size_t dataSize, cl_mem gpuBuffer)
        {
            cl_int err = clEnqueueReadBuffer(
                commandQueue(), gpuBuffer, CL_TRUE, 0, dataSize, dst, 0, nullptr, nullptr);
            if (err)
                throw OCL_EXCEPTION;
        }
    };

    int alignedSize(int len)
    {
        return SimdOpenCl::alignedSize(len);
    }

} // internals

namespace arithmetic
{
    _SIMD_OCL_SPEC void addC(const float * pSrc, float val, float * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::AddC_32f>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void subC(const float * pSrc, float val, float * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::SubC_32f>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void mulC(const float * pSrc, float val, float * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::MulC_32f>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void divC(const float * pSrc, float val, float * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::DivC_32f>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void subCRev(const float * pSrc, float val, float * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::SubCRev_32f>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void divCRev(const float * pSrc, float val, float * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::DivCRev_32f>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void add(const float * pSrc1, const float * pSrc2, float * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::Add_32f>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void sub(const float * pSrc1, const float * pSrc2, float * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::Sub_32f>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void mul(const float * pSrc1, const float * pSrc2, float * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::Mul_32f>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void div(const float * pSrc1, const float * pSrc2, float * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::Div_32f>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void abs(const float* pSrc, float* pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::Abs_32f>(pSrc, 0.0f, pDst, len);
    }
}

} // ocl
