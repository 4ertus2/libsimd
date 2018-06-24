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

#define OCL_EXCEPTION(err) simd::Exception(__FILE__, __LINE__, __FUNCTION__, err)
#define OCL_EXCEPT(err, msg) simd::Exception(__FILE__, __LINE__, __FUNCTION__, err, msg)

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
                throw OCL_EXCEPTION(err);

            dst_ = std::shared_ptr<ClMemType>(
                clCreateBuffer(gpuContext, CL_MEM_WRITE_ONLY, dataSize(), nullptr, &err), clReleaseMemObject);
            if (err)
                throw OCL_EXCEPTION(err);
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
                throw OCL_EXCEPTION(err);
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
            cl_int err = 0;
            err = clGetPlatformIDs(0, nullptr, &numPlatforms);
            if (err || numPlatforms <= platformId)
                throw OCL_EXCEPTION(err);

            std::vector<cl_platform_id> platforms(numPlatforms);
            err = clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
            if (err)
                throw OCL_EXCEPTION(err);
            platform_ = platforms[platformId];

            cl_uint numDevices;
            err = clGetDeviceIDs(platform_, deviceType, 0, nullptr, &numDevices);
            if (err || numDevices <= deviceId)
                throw OCL_EXCEPTION(err);
            std::vector<cl_device_id> devices(numDevices);
            err = clGetDeviceIDs(platform_, deviceType, numDevices, devices.data(), nullptr);
            if (err)
                throw OCL_EXCEPTION(err);
            device_ = devices[deviceId];

            gpuContext_ = std::shared_ptr<ContextType>(
                clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err), clReleaseContext);
            if (err)
                throw OCL_EXCEPTION(err);

#ifdef CL_USE_DEPRECATED_OPENCL_1_2_APIS
            commandQueue_ = std::shared_ptr<CmdQueueType>(
                clCreateCommandQueue(gpuContext(), device_, 0, &err), clReleaseCommandQueue);
#else
            commandQueue_ = std::shared_ptr<CmdQueueType>(
                clCreateCommandQueueWithProperties(gpuContext(), device_, nullptr, &err), clReleaseCommandQueue);
#endif
            if (err)
                throw OCL_EXCEPTION(err);

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
                throw OCL_EXCEPTION(0);

            prog.setProgram(clCreateProgramWithSource(gpuContext(), 1, (const char **)&srcProg.text, &progLength, &err));
            if (err)
                throw OCL_EXCEPTION(err);

            //const char* flags = "-cl-fast-relaxed-math";
            err = clBuildProgram(prog.program(), 1, &device_, nullptr, nullptr, nullptr);
            if (err == CL_BUILD_PROGRAM_FAILURE)
            {
                size_t size;
                err = clGetProgramBuildInfo(prog.program(), device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &size);
                if (err)
                    OCL_EXCEPTION(err);
                std::vector<char> info(size);
                clGetProgramBuildInfo(prog.program(), device_, CL_PROGRAM_BUILD_LOG, size, info.data(), nullptr);
                if (err)
                    OCL_EXCEPTION(err);
                OCL_EXCEPT(CL_BUILD_PROGRAM_FAILURE, info.data());
            }
            if (err)
                throw OCL_EXCEPTION(err);

            prog.setKernel(clCreateKernel(prog.program(), srcProg.name, &err));
            if (err)
                throw OCL_EXCEPTION(err);
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
                throw OCL_EXCEPTION(err);
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
                throw OCL_EXCEPTION(err);
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
                throw OCL_EXCEPTION(err);

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
                throw OCL_EXCEPTION(err);
        }

        void asyncWriteToGPU(const void * src, size_t dataSize, cl_mem gpuBuffer)
        {
            cl_int err = clEnqueueWriteBuffer(
                commandQueue(), gpuBuffer, CL_FALSE, 0, dataSize, src, 0, nullptr, nullptr);
            if (err)
                throw OCL_EXCEPTION(err);
        }

        void syncReadFromGPU(void * dst, size_t dataSize, cl_mem gpuBuffer)
        {
            cl_int err = clEnqueueReadBuffer(
                commandQueue(), gpuBuffer, CL_TRUE, 0, dataSize, dst, 0, nullptr, nullptr);
            if (err)
                throw OCL_EXCEPTION(err);
        }
    };

    int alignedSize(int len)
    {
        return SimdOpenCl::alignedSize(len);
    }

    template <typename _KernelT, typename _T>
    void execKernel(const _T * pSrc1, const _T * pSrc2, _T * pDst, int len)
    {
        SimdOpenCl::getInstance().exec<_KernelT>(pSrc1, pSrc2, pDst, len);
    }

    template <typename _KernelT, typename _T>
    void execKernel(const _T * pSrc, _T val, _T * pDst, int len)
    {
        SimdOpenCl::getInstance().exec<_KernelT>(pSrc, val, pDst, len);
    }

} // internals

namespace arithmetic
{
    // int8_t

    _SIMD_OCL_SPEC void addC(const int8_t * pSrc, int8_t val, int8_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::AddC<int8_t>>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void subC(const int8_t * pSrc, int8_t val, int8_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::SubC<int8_t>>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void mulC(const int8_t * pSrc, int8_t val, int8_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::MulC<int8_t>>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void divC(const int8_t * pSrc, int8_t val, int8_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::DivC<int8_t>>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void subCRev(const int8_t * pSrc, int8_t val, int8_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::SubCRev<int8_t>>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void divCRev(const int8_t * pSrc, int8_t val, int8_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::DivCRev<int8_t>>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void add(const int8_t * pSrc1, const int8_t * pSrc2, int8_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::Add<int8_t>>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void sub(const int8_t * pSrc1, const int8_t * pSrc2, int8_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::Sub<int8_t>>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void mul(const int8_t * pSrc1, const int8_t * pSrc2, int8_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::Mul<int8_t>>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void div(const int8_t * pSrc1, const int8_t * pSrc2, int8_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::Div<int8_t>>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void abs(const int8_t * pSrc, int8_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::Abs<int8_t>>(pSrc, int8_t(0), pDst, len);
    }

    // uint16_t

    _SIMD_OCL_SPEC void addC(const uint8_t * pSrc, uint8_t val, uint8_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::AddC<uint8_t>>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void subC(const uint8_t * pSrc, uint8_t val, uint8_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::SubC<uint8_t>>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void mulC(const uint8_t * pSrc, uint8_t val, uint8_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::MulC<uint8_t>>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void divC(const uint8_t * pSrc, uint8_t val, uint8_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::DivC<uint8_t>>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void subCRev(const uint8_t * pSrc, uint8_t val, uint8_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::SubCRev<uint8_t>>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void divCRev(const uint8_t * pSrc, uint8_t val, uint8_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::DivCRev<uint8_t>>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void add(const uint8_t * pSrc1, const uint8_t * pSrc2, uint8_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::Add<uint8_t>>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void sub(const uint8_t * pSrc1, const uint8_t * pSrc2, uint8_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::Sub<uint8_t>>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void mul(const uint8_t * pSrc1, const uint8_t * pSrc2, uint8_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::Mul<uint8_t>>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void div(const uint8_t * pSrc1, const uint8_t * pSrc2, uint8_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::Div<uint8_t>>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void abs(const uint8_t * pSrc, uint8_t * pDst, int len)
    {
        if (pSrc != pDst)
            copy(pSrc, pDst, len);
    }

    // int16_t

    _SIMD_OCL_SPEC void addC(const int16_t * pSrc, int16_t val, int16_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::AddC<int16_t>>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void subC(const int16_t * pSrc, int16_t val, int16_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::SubC<int16_t>>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void mulC(const int16_t * pSrc, int16_t val, int16_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::MulC<int16_t>>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void divC(const int16_t * pSrc, int16_t val, int16_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::DivC<int16_t>>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void subCRev(const int16_t * pSrc, int16_t val, int16_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::SubCRev<int16_t>>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void divCRev(const int16_t * pSrc, int16_t val, int16_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::DivCRev<int16_t>>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void add(const int16_t * pSrc1, const int16_t * pSrc2, int16_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::Add<int16_t>>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void sub(const int16_t * pSrc1, const int16_t * pSrc2, int16_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::Sub<int16_t>>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void mul(const int16_t * pSrc1, const int16_t * pSrc2, int16_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::Mul<int16_t>>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void div(const int16_t * pSrc1, const int16_t * pSrc2, int16_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::Div<int16_t>>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void abs(const int16_t * pSrc, int16_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::Abs<int16_t>>(pSrc, int16_t(0), pDst, len);
    }

    // uint16_t

    _SIMD_OCL_SPEC void addC(const uint16_t * pSrc, uint16_t val, uint16_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::AddC<uint16_t>>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void subC(const uint16_t * pSrc, uint16_t val, uint16_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::SubC<uint16_t>>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void mulC(const uint16_t * pSrc, uint16_t val, uint16_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::MulC<uint16_t>>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void divC(const uint16_t * pSrc, uint16_t val, uint16_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::DivC<uint16_t>>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void subCRev(const uint16_t * pSrc, uint16_t val, uint16_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::SubCRev<uint16_t>>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void divCRev(const uint16_t * pSrc, uint16_t val, uint16_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::DivCRev<uint16_t>>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void add(const uint16_t * pSrc1, const uint16_t * pSrc2, uint16_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::Add<uint16_t>>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void sub(const uint16_t * pSrc1, const uint16_t * pSrc2, uint16_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::Sub<uint16_t>>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void mul(const uint16_t * pSrc1, const uint16_t * pSrc2, uint16_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::Mul<uint16_t>>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void div(const uint16_t * pSrc1, const uint16_t * pSrc2, uint16_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::Div<uint16_t>>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void abs(const uint16_t * pSrc, uint16_t * pDst, int len)
    {
        if (pSrc != pDst)
            copy(pSrc, pDst, len);
    }

    // int32_t

    _SIMD_OCL_SPEC void addC(const int32_t * pSrc, int32_t val, int32_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::AddC<int32_t>>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void subC(const int32_t * pSrc, int32_t val, int32_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::SubC<int32_t>>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void mulC(const int32_t * pSrc, int32_t val, int32_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::MulC<int32_t>>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void divC(const int32_t * pSrc, int32_t val, int32_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::DivC<int32_t>>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void subCRev(const int32_t * pSrc, int32_t val, int32_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::SubCRev<int32_t>>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void divCRev(const int32_t * pSrc, int32_t val, int32_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::DivCRev<int32_t>>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void add(const int32_t * pSrc1, const int32_t * pSrc2, int32_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::Add<int32_t>>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void sub(const int32_t * pSrc1, const int32_t * pSrc2, int32_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::Sub<int32_t>>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void mul(const int32_t * pSrc1, const int32_t * pSrc2, int32_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::Mul<int32_t>>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void div(const int32_t * pSrc1, const int32_t * pSrc2, int32_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::Div<int32_t>>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void abs(const int32_t * pSrc, int32_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::Abs<int32_t>>(pSrc, 0, pDst, len);
    }

    // uint32_t

    _SIMD_OCL_SPEC void addC(const uint32_t * pSrc, uint32_t val, uint32_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::AddC<uint32_t>>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void subC(const uint32_t * pSrc, uint32_t val, uint32_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::SubC<uint32_t>>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void mulC(const uint32_t * pSrc, uint32_t val, uint32_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::MulC<uint32_t>>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void divC(const uint32_t * pSrc, uint32_t val, uint32_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::DivC<uint32_t>>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void subCRev(const uint32_t * pSrc, uint32_t val, uint32_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::SubCRev<uint32_t>>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void divCRev(const uint32_t * pSrc, uint32_t val, uint32_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::DivCRev<uint32_t>>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void add(const uint32_t * pSrc1, const uint32_t * pSrc2, uint32_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::Add<uint32_t>>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void sub(const uint32_t * pSrc1, const uint32_t * pSrc2, uint32_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::Sub<uint32_t>>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void mul(const uint32_t * pSrc1, const uint32_t * pSrc2, uint32_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::Mul<uint32_t>>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void div(const uint32_t * pSrc1, const uint32_t * pSrc2, uint32_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::Div<uint32_t>>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void abs(const uint32_t * pSrc, uint32_t * pDst, int len)
    {
        if (pSrc != pDst)
            copy(pSrc, pDst, len);
    }

    // int64_t

    _SIMD_OCL_SPEC void addC(const int64_t * pSrc, int64_t val, int64_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::AddC<int64_t>>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void subC(const int64_t * pSrc, int64_t val, int64_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::SubC<int64_t>>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void mulC(const int64_t * pSrc, int64_t val, int64_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::MulC<int64_t>>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void divC(const int64_t * pSrc, int64_t val, int64_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::DivC<int64_t>>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void subCRev(const int64_t * pSrc, int64_t val, int64_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::SubCRev<int64_t>>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void divCRev(const int64_t * pSrc, int64_t val, int64_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::DivCRev<int64_t>>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void add(const int64_t * pSrc1, const int64_t * pSrc2, int64_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::Add<int64_t>>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void sub(const int64_t * pSrc1, const int64_t * pSrc2, int64_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::Sub<int64_t>>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void mul(const int64_t * pSrc1, const int64_t * pSrc2, int64_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::Mul<int64_t>>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void div(const int64_t * pSrc1, const int64_t * pSrc2, int64_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::Div<int64_t>>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void abs(const int64_t * pSrc, int64_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::Abs<int64_t>>(pSrc, 0l, pDst, len);
    }

    // uint64_t

    _SIMD_OCL_SPEC void addC(const uint64_t * pSrc, uint64_t val, uint64_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::AddC<uint64_t>>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void subC(const uint64_t * pSrc, uint64_t val, uint64_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::SubC<uint64_t>>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void mulC(const uint64_t * pSrc, uint64_t val, uint64_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::MulC<uint64_t>>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void divC(const uint64_t * pSrc, uint64_t val, uint64_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::DivC<uint64_t>>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void subCRev(const uint64_t * pSrc, uint64_t val, uint64_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::SubCRev<uint64_t>>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void divCRev(const uint64_t * pSrc, uint64_t val, uint64_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::DivCRev<uint64_t>>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void add(const uint64_t * pSrc1, const uint64_t * pSrc2, uint64_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::Add<uint64_t>>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void sub(const uint64_t * pSrc1, const uint64_t * pSrc2, uint64_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::Sub<uint64_t>>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void mul(const uint64_t * pSrc1, const uint64_t * pSrc2, uint64_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::Mul<uint64_t>>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void div(const uint64_t * pSrc1, const uint64_t * pSrc2, uint64_t * pDst, int len)
    {
        internals::execKernel<internals::Kernel::Div<uint64_t>>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void abs(const uint64_t * pSrc, uint64_t * pDst, int len)
    {
        if (pSrc != pDst)
            copy(pSrc, pDst, len);
    }

    // float

    _SIMD_OCL_SPEC void addC(const float * pSrc, float val, float * pDst, int len)
    {
        internals::execKernel<internals::Kernel::AddC<float>>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void subC(const float * pSrc, float val, float * pDst, int len)
    {
        internals::execKernel<internals::Kernel::SubC<float>>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void mulC(const float * pSrc, float val, float * pDst, int len)
    {
        internals::execKernel<internals::Kernel::MulC<float>>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void divC(const float * pSrc, float val, float * pDst, int len)
    {
        internals::execKernel<internals::Kernel::DivC<float>>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void subCRev(const float * pSrc, float val, float * pDst, int len)
    {
        internals::execKernel<internals::Kernel::SubCRev<float>>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void divCRev(const float * pSrc, float val, float * pDst, int len)
    {
        internals::execKernel<internals::Kernel::DivCRev<float>>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void add(const float * pSrc1, const float * pSrc2, float * pDst, int len)
    {
        internals::execKernel<internals::Kernel::Add<float>>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void sub(const float * pSrc1, const float * pSrc2, float * pDst, int len)
    {
        internals::execKernel<internals::Kernel::Sub<float>>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void mul(const float * pSrc1, const float * pSrc2, float * pDst, int len)
    {
        internals::execKernel<internals::Kernel::Mul<float>>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void div(const float * pSrc1, const float * pSrc2, float * pDst, int len)
    {
        internals::execKernel<internals::Kernel::Div<float>>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void abs(const float * pSrc, float * pDst, int len)
    {
        internals::execKernel<internals::Kernel::Abs<float>>(pSrc, 0.0f, pDst, len);
    }

    // double

    _SIMD_OCL_SPEC void addC(const double * pSrc, double val, double * pDst, int len)
    {
        internals::execKernel<internals::Kernel::AddC<double>>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void subC(const double * pSrc, double val, double * pDst, int len)
    {
        internals::execKernel<internals::Kernel::SubC<double>>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void mulC(const double * pSrc, double val, double * pDst, int len)
    {
        internals::execKernel<internals::Kernel::MulC<double>>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void divC(const double * pSrc, double val, double * pDst, int len)
    {
        internals::execKernel<internals::Kernel::DivC<double>>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void subCRev(const double * pSrc, double val, double * pDst, int len)
    {
        internals::execKernel<internals::Kernel::SubCRev<double>>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void divCRev(const double * pSrc, double val, double * pDst, int len)
    {
        internals::execKernel<internals::Kernel::DivCRev<double>>(pSrc, val, pDst, len);
    }

    _SIMD_OCL_SPEC void add(const double * pSrc1, const double * pSrc2, double * pDst, int len)
    {
        internals::execKernel<internals::Kernel::Add<double>>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void sub(const double * pSrc1, const double * pSrc2, double * pDst, int len)
    {
        internals::execKernel<internals::Kernel::Sub<double>>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void mul(const double * pSrc1, const double * pSrc2, double * pDst, int len)
    {
        internals::execKernel<internals::Kernel::Mul<double>>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void div(const double * pSrc1, const double * pSrc2, double * pDst, int len)
    {
        internals::execKernel<internals::Kernel::Div<double>>(pSrc1, pSrc2, pDst, len);
    }

    _SIMD_OCL_SPEC void abs(const double * pSrc, double * pDst, int len)
    {
        internals::execKernel<internals::Kernel::Abs<double>>(pSrc, 0.0, pDst, len);
    }
}

} // ocl
