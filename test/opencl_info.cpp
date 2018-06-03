#include <vector>
#include <string>
#include <iostream>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

static const char * toString(const cl_device_type& devType)
{
    if (devType == CL_DEVICE_TYPE_DEFAULT)
        return "CL_DEVICE_TYPE_DEFAULT";
    if (devType == CL_DEVICE_TYPE_CPU)
        return "CL_DEVICE_TYPE_CPU";
    if (devType == CL_DEVICE_TYPE_GPU)
        return "CL_DEVICE_TYPE_GPU";
    if (devType == CL_DEVICE_TYPE_ACCELERATOR)
        return "CL_DEVICE_TYPE_ACCELERATOR";
    if (devType == CL_DEVICE_TYPE_CUSTOM)
        return "CL_DEVICE_TYPE_CUSTOM";
    return "UNKNOWN";
}

static std::string fpToString(const cl_device_fp_config& fpconf)
{
    std::string out;
    if (fpconf & CL_FP_DENORM)
        out += "DENORM ";
    if (fpconf & CL_FP_INF_NAN)
        out += "INF_NAN ";
    if (fpconf & CL_FP_ROUND_TO_NEAREST)
        out += "ROUND_TO_NEAREST ";
    if (fpconf & CL_FP_ROUND_TO_ZERO)
        out += "ROUND_TO_ZERO ";
    if (fpconf & CL_FP_ROUND_TO_INF)
        out += "ROUND_TO_INF ";
    if (fpconf & CL_FP_FMA)
        out += "FMA ";
    if (fpconf & CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT)
        out += "CORRECTLY_ROUNDED_DIVIDE_SQRT ";
    if (fpconf & CL_FP_SOFT_FLOAT)
        out += "SOFT_FLOAT ";
    return out;
}

static void openClInfo(uint32_t infoBufSize = 1024 * 1024)
{
    cl_uint numPlatforms;
    if (clGetPlatformIDs(0, nullptr, &numPlatforms))
        throw __LINE__;
    std::cout << "platforms count: " << numPlatforms << std::endl << std::endl;

    std::vector<cl_platform_id> platforms(numPlatforms);
    if (clGetPlatformIDs(numPlatforms, platforms.data(), nullptr))
        throw __LINE__;

    std::vector<char> info(infoBufSize);
    for (auto& pl : platforms)
    {
        if (clGetPlatformInfo(pl, CL_PLATFORM_PROFILE, infoBufSize, info.data(), nullptr))
            throw __LINE__;
        std::string profile(info.data());

        if (clGetPlatformInfo(pl, CL_PLATFORM_VERSION, infoBufSize, info.data(), nullptr))
            throw __LINE__;
        std::string version(info.data());

        if (clGetPlatformInfo(pl, CL_PLATFORM_NAME, infoBufSize, info.data(), nullptr))
            throw __LINE__;
        std::string name(info.data());

        if (clGetPlatformInfo(pl, CL_PLATFORM_VENDOR, infoBufSize, info.data(), nullptr))
            throw __LINE__;
        std::string vendor(info.data());

        if (clGetPlatformInfo(pl, CL_PLATFORM_EXTENSIONS, infoBufSize, info.data(), nullptr))
            throw __LINE__;
        std::string extensions(info.data());

        std::cout << "[p] profile: " << profile << std::endl
            << "[p] version: " << version << std::endl
            << "[p] name: " << name << std::endl
            << "[p] vendor: " << vendor << std::endl
            << "[p] extensions: " << extensions << std::endl;

        cl_uint numDevices;
        if (clGetDeviceIDs(pl, CL_DEVICE_TYPE_ALL, 0, nullptr, &numDevices))
            throw __LINE__;
        std::cout << "[p] devies: " << numDevices << std::endl << std::endl;

        std::vector<cl_device_id> devices(numDevices);
        if (clGetDeviceIDs(pl, CL_DEVICE_TYPE_ALL, numDevices, devices.data(), nullptr))
            throw __LINE__;
        for (auto& dev : devices)
        {
            cl_device_type type;
            if (clGetDeviceInfo(dev, CL_DEVICE_TYPE, sizeof(cl_device_type), &type, nullptr))
                throw __LINE__;
            std::cout << "[d] device type: " << toString(type) << std::endl;

            if (clGetDeviceInfo(dev, CL_DEVICE_PROFILE, infoBufSize, info.data(), nullptr))
                throw __LINE__;
            std::cout << "[d] profile: " << std::string(info.data()) << std::endl;

            if (clGetDeviceInfo(dev, CL_DEVICE_OPENCL_C_VERSION, infoBufSize, info.data(), nullptr))
                throw __LINE__;
            std::cout << "[d] opencl ver: " << std::string(info.data()) << std::endl;

            if (clGetDeviceInfo(dev, CL_DEVICE_NAME, infoBufSize, info.data(), nullptr))
                throw __LINE__;
            std::cout << "[d] name: " << std::string(info.data()) << std::endl;

            if (clGetDeviceInfo(dev, CL_DEVICE_VENDOR, infoBufSize, info.data(), nullptr))
                throw __LINE__;
            std::cout << "[d] vendor: " << std::string(info.data()) << std::endl;

            if (clGetDeviceInfo(dev, CL_DEVICE_VERSION, infoBufSize, info.data(), nullptr))
                throw __LINE__;
            std::cout << "[d] version: " << std::string(info.data()) << std::endl;

            if (clGetDeviceInfo(dev, CL_DRIVER_VERSION, infoBufSize, info.data(), nullptr))
                throw __LINE__;
            std::cout << "[d] driver version: " << std::string(info.data()) << std::endl;

            if (clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, infoBufSize, info.data(), nullptr))
                throw __LINE__;
            std::cout << "[d] extensions: " << std::string(info.data()) << std::endl;

            cl_uint value;
            if (clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &value, nullptr))
                throw __LINE__;
            std::cout << "[d] max compute units: " << value << std::endl;

            if (clGetDeviceInfo(dev, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &value, nullptr))
                throw __LINE__;
            std::cout << "[d] max clock frequency: " << value << " MHz" << std::endl;

            if (clGetDeviceInfo(dev, CL_DEVICE_ADDRESS_BITS, sizeof(cl_uint), &value, nullptr))
                throw __LINE__;
            std::cout << "[d] address bits: " << value << std::endl;

            cl_ulong longValue;
            if (clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &longValue, nullptr))
                throw __LINE__;
            std::cout << "[d] global mem size: " << longValue / (1024 * 1024) << " Mb" << std::endl;

            if (clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(cl_ulong), &longValue, nullptr))
                throw __LINE__;
            std::cout << "[d] global mem cache size: " << longValue << std::endl;

            if (clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, sizeof(cl_uint), &value, nullptr))
                throw __LINE__;
            std::cout << "[d] global mem cacheline size: " << value << std::endl;

            if (clGetDeviceInfo(dev, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &longValue, nullptr))
                throw __LINE__;
            std::cout << "[d] local mem size: " << longValue << std::endl;

            if (clGetDeviceInfo(dev, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(cl_uint), &value, nullptr))
                throw __LINE__;
            std::cout << "[d] base addr align: " << value << std::endl;

            if (clGetDeviceInfo(dev, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &longValue, nullptr))
                throw __LINE__;
            std::cout << "[d] max mem alloc size: " << longValue / (1024 * 1024) << " Mb" << std::endl;

            if (clGetDeviceInfo(dev, CL_DEVICE_MAX_CONSTANT_ARGS, sizeof(cl_uint), &value, nullptr))
                throw __LINE__;
            std::cout << "[d] max constant args: " << value << std::endl;

            if (clGetDeviceInfo(dev, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(cl_ulong), &longValue, nullptr))
                throw __LINE__;
            std::cout << "[d] max constant buffer size: " << longValue / 1024 << " Kb" << std::endl;

            size_t num;
            if (clGetDeviceInfo(dev, CL_DEVICE_MAX_PARAMETER_SIZE, sizeof(size_t), &num, nullptr))
                throw __LINE__;
            std::cout << "[d] max parameter size: " << num << std::endl;

            if (clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &num, nullptr))
                throw __LINE__;
            std::cout << "[d] max work group size: " << num << std::endl;

            cl_uint numDims;
            if (clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &numDims, nullptr))
                throw __LINE__;
            std::cout << "[d] max work group item dimensions: " << numDims << std::endl;

            std::vector<size_t> dims(numDims);
            if (clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t)*numDims, dims.data(), nullptr))
                throw __LINE__;
            std::cout << "[d] max work item sizes: ( ";
            for (size_t& dim : dims)
                std::cout << dim << " ";
            std::cout << ")" << std::endl;

            if (clGetDeviceInfo(dev, CL_DEVICE_BUILT_IN_KERNELS, infoBufSize, info.data(), nullptr))
                throw __LINE__;
            std::cout << "[d] builtin kernels: " << std::string(info.data()) << std::endl;

            cl_device_fp_config fpconf;
            if (clGetDeviceInfo(dev, CL_DEVICE_SINGLE_FP_CONFIG, sizeof(cl_device_fp_config), &fpconf, nullptr))
                throw __LINE__;
            std::cout << "[d] single fp config: " << fpToString(fpconf) << std::endl;

            if (clGetDeviceInfo(dev, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(cl_device_fp_config), &fpconf, nullptr))
                throw __LINE__;
            std::cout << "[d] double fp config: " << fpToString(fpconf) << std::endl;

            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

int main()
{
    try
    {
        openClInfo();
    }
    catch (const int& line)
    {
        std::cerr << "line: " << line << std::endl;
        return 1;
    }

    return 0;
}
