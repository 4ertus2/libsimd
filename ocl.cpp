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
        internals::SimdOpenCl::getInstance().exec<_KernelT>(pSrc1, pSrc2, pDst, len);
    }

    template <typename _KernelT, typename _T>
    void execKernel(const _T * pSrc, _T val, _T * pDst, int len)
    {
        internals::SimdOpenCl::getInstance().exec<_KernelT>(pSrc, val, pDst, len);
    }

} // internals

namespace arithmetic
{
    template <typename _T>
    void regFuncs()
    {
        const _T * pSrc = nullptr;
        const _T * pSrc1 = nullptr;
        const _T * pSrc2 = nullptr;
        _T val(0);
        _T * pDst = nullptr;
        int len = 0;

        internals::execKernel<internals::Kernel::Add<_T>>(pSrc1, pSrc2, pDst, len);
        internals::execKernel<internals::Kernel::Sub<_T>>(pSrc1, pSrc2, pDst, len);
        internals::execKernel<internals::Kernel::Mul<_T>>(pSrc1, pSrc2, pDst, len);
        internals::execKernel<internals::Kernel::Div<_T>>(pSrc1, pSrc2, pDst, len);

        internals::execKernel<internals::Kernel::AddC<_T>>(pSrc, val, pDst, len);
        internals::execKernel<internals::Kernel::SubC<_T>>(pSrc, val, pDst, len);
        internals::execKernel<internals::Kernel::MulC<_T>>(pSrc, val, pDst, len);
        internals::execKernel<internals::Kernel::DivC<_T>>(pSrc, val, pDst, len);
        internals::execKernel<internals::Kernel::SubCRev<_T>>(pSrc, val, pDst, len);
        internals::execKernel<internals::Kernel::DivCRev<_T>>(pSrc, val, pDst, len);
    }

    void registerFunctions()
    {
        regFuncs<int8_t>();
        regFuncs<uint8_t>();
        regFuncs<int16_t>();
        regFuncs<uint16_t>();
        regFuncs<int32_t>();
        regFuncs<uint32_t>();
        regFuncs<int64_t>();
        regFuncs<uint64_t>();
        regFuncs<float>();
        regFuncs<double>();

        int len = 0;
        int8_t i8 = 0;
        int16_t i16 = 0;
        int32_t i32 = 0;
        int64_t i64 = 0;
        float f32 = 0;
        double f64 = 0;
        internals::execKernel<internals::Kernel::Abs<int8_t>>((const int8_t*)&i8, i8, &i8, len);
        internals::execKernel<internals::Kernel::Abs<int16_t>>((const int16_t*)&i16, i16, &i16, len);
        internals::execKernel<internals::Kernel::Abs<int32_t>>((const int32_t*)&i32, i32, &i32, len);
        internals::execKernel<internals::Kernel::Abs<int64_t>>((const int64_t*)&i64, i64, &i64, len);
        internals::execKernel<internals::Kernel::Abs<float>>((const float*)&f32, f32, &f32, len);
        internals::execKernel<internals::Kernel::Abs<double>>((const double*)&f64, f64, &f64, len);
    }
}

} // ocl
