#include "ocl.h"
#include "ocl_kernels.h"

#include <vector>
#include <cstring>
#include <memory>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define OCL_EXCEPTION simd::Exception(__FILE__, __LINE__, __FUNCTION__)

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

    template <typename _T>
    struct BufferType;

    template <typename _T>
    struct BufferType<Kernel::PtrPtrPtr<_T>>
    {
        using Type = SrcSrcDstBuffers<_T>;
    };

    template <typename _T>
    struct BufferType<Kernel::PtrValPtr<_T>>
    {
        using Type = SrcValDstBuffers<_T>;
    };

    ///
    class SimdOpenCl
    {
    public:
        static SimdOpenCl& getInstance()
        {
            static const uint32_t platformId = 0;
            static const uint32_t deviceId = 0;
            static SimdOpenCl instance(platformId, CL_DEVICE_TYPE_GPU, deviceId);
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
            typename BufferType<typename _T::BaseType>::Type buffers(gpuContext(), workSize);
            //size_t dataSize = buffers.dataSize();

            cl_kernel kernel = programs_[static_cast<uint32_t>(_T::id())].kernel();
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

        SimdOpenCl(uint32_t platformId, cl_device_type deviceType, uint32_t deviceId)
        {
            cl_uint numPlatforms;
            if (clGetPlatformIDs(0, nullptr, &numPlatforms) || numPlatforms <= platformId)
                throw OCL_EXCEPTION;

            std::vector<cl_platform_id> platforms(numPlatforms);
            if (clGetPlatformIDs(numPlatforms, platforms.data(), nullptr))
                throw OCL_EXCEPTION;
            platform_ = platforms[platformId];

            cl_uint numDevices;
            if (clGetDeviceIDs(platform_, deviceType, 0, nullptr, &numDevices) || numDevices <= deviceId)
                throw OCL_EXCEPTION;
            std::vector<cl_device_id> devices(numDevices);
            if (clGetDeviceIDs(platform_, deviceType, numDevices, devices.data(), nullptr))
                throw OCL_EXCEPTION;
            device_ = devices[deviceId];

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
            Kernel::TextProgram srcProg = Kernel::program(func);
            size_t progLength = strlen(srcProg.text);

            if (programs_.size() <= static_cast<uint32_t>(func))
                throw OCL_EXCEPTION;

            prog.setProgram(clCreateProgramWithSource(gpuContext(), 1, (const char **)&srcProg.text, &progLength, &err));
            if (err)
                throw OCL_EXCEPTION;

            //const char* flags = "-cl-fast-relaxed-math";
            err = clBuildProgram(prog.program(), 0, nullptr, nullptr, nullptr, nullptr);
            if (err)
                throw OCL_EXCEPTION;

            prog.setKernel(clCreateKernel(prog.program(), srcProg.name, &err));
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
    // int32_t

    _SIMD_OCL_SPEC void addC(const int32_t * pSrc, int32_t val, int32_t * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::AddC_32s>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void subC(const int32_t * pSrc, int32_t val, int32_t * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::SubC_32s>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void mulC(const int32_t * pSrc, int32_t val, int32_t * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::MulC_32s>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void divC(const int32_t * pSrc, int32_t val, int32_t * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::DivC_32s>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void subCRev(const int32_t * pSrc, int32_t val, int32_t * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::SubCRev_32s>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void divCRev(const int32_t * pSrc, int32_t val, int32_t * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::DivCRev_32s>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void add(const int32_t * pSrc1, const int32_t * pSrc2, int32_t * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::Add_32s>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void sub(const int32_t * pSrc1, const int32_t * pSrc2, int32_t * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::Sub_32s>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void mul(const int32_t * pSrc1, const int32_t * pSrc2, int32_t * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::Mul_32s>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void div(const int32_t * pSrc1, const int32_t * pSrc2, int32_t * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::Div_32s>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void abs(const int32_t * pSrc, int32_t * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::Abs_32s>(pSrc, 0.0f, pDst, len);
    }

    // uint32_t

    _SIMD_OCL_SPEC void addC(const uint32_t * pSrc, uint32_t val, uint32_t * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::AddC_32u>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void subC(const uint32_t * pSrc, uint32_t val, uint32_t * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::SubC_32u>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void mulC(const uint32_t * pSrc, uint32_t val, uint32_t * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::MulC_32u>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void divC(const uint32_t * pSrc, uint32_t val, uint32_t * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::DivC_32u>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void subCRev(const uint32_t * pSrc, uint32_t val, uint32_t * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::SubCRev_32u>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void divCRev(const uint32_t * pSrc, uint32_t val, uint32_t * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::DivCRev_32u>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void add(const uint32_t * pSrc1, const uint32_t * pSrc2, uint32_t * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::Add_32u>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void sub(const uint32_t * pSrc1, const uint32_t * pSrc2, uint32_t * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::Sub_32u>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void mul(const uint32_t * pSrc1, const uint32_t * pSrc2, uint32_t * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::Mul_32u>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void div(const uint32_t * pSrc1, const uint32_t * pSrc2, uint32_t * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::Div_32u>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void abs(const uint32_t * pSrc, uint32_t * pDst, int len)
    {
        if (pSrc != pDst)
            copy(pSrc, pDst, len);
    }

    // int64_t

    _SIMD_OCL_SPEC void addC(const int64_t * pSrc, int64_t val, int64_t * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::AddC_64s>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void subC(const int64_t * pSrc, int64_t val, int64_t * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::SubC_64s>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void mulC(const int64_t * pSrc, int64_t val, int64_t * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::MulC_64s>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void divC(const int64_t * pSrc, int64_t val, int64_t * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::DivC_64s>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void subCRev(const int64_t * pSrc, int64_t val, int64_t * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::SubCRev_64s>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void divCRev(const int64_t * pSrc, int64_t val, int64_t * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::DivCRev_64s>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void add(const int64_t * pSrc1, const int64_t * pSrc2, int64_t * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::Add_64s>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void sub(const int64_t * pSrc1, const int64_t * pSrc2, int64_t * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::Sub_64s>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void mul(const int64_t * pSrc1, const int64_t * pSrc2, int64_t * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::Mul_64s>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void div(const int64_t * pSrc1, const int64_t * pSrc2, int64_t * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::Div_64s>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void abs(const int64_t * pSrc, int64_t * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::Abs_64s>(pSrc, 0.0f, pDst, len);
    }

    // uint64_t

    _SIMD_OCL_SPEC void addC(const uint64_t * pSrc, uint64_t val, uint64_t * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::AddC_64u>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void subC(const uint64_t * pSrc, uint64_t val, uint64_t * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::SubC_64u>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void mulC(const uint64_t * pSrc, uint64_t val, uint64_t * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::MulC_64u>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void divC(const uint64_t * pSrc, uint64_t val, uint64_t * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::DivC_64u>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void subCRev(const uint64_t * pSrc, uint64_t val, uint64_t * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::SubCRev_64u>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void divCRev(const uint64_t * pSrc, uint64_t val, uint64_t * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::DivCRev_64u>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void add(const uint64_t * pSrc1, const uint64_t * pSrc2, uint64_t * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::Add_64u>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void sub(const uint64_t * pSrc1, const uint64_t * pSrc2, uint64_t * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::Sub_64u>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void mul(const uint64_t * pSrc1, const uint64_t * pSrc2, uint64_t * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::Mul_64u>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void div(const uint64_t * pSrc1, const uint64_t * pSrc2, uint64_t * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::Div_64u>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void abs(const uint64_t * pSrc, uint64_t * pDst, int len)
    {
        if (pSrc != pDst)
            copy(pSrc, pDst, len);
    }

    // float

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

    _SIMD_OCL_SPEC void abs(const float * pSrc, float * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::Abs_32f>(pSrc, 0.0f, pDst, len);
    }

    // double

    _SIMD_OCL_SPEC void addC(const double * pSrc, double val, double * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::AddC_64f>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void subC(const double * pSrc, double val, double * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::SubC_64f>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void mulC(const double * pSrc, double val, double * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::MulC_64f>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void divC(const double * pSrc, double val, double * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::DivC_64f>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void subCRev(const double * pSrc, double val, double * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::SubCRev_64f>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void divCRev(const double * pSrc, double val, double * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::DivCRev_64f>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void add(const double * pSrc1, const double * pSrc2, double * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::Add_64f>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void sub(const double * pSrc1, const double * pSrc2, double * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::Sub_64f>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void mul(const double * pSrc1, const double * pSrc2, double * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::Mul_64f>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void div(const double * pSrc1, const double * pSrc2, double * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::Div_64f>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void abs(const double * pSrc, double * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<internals::Kernel::Abs_64f>(pSrc, 0.0f, pDst, len);
    }
}

} // ocl
