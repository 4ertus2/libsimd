cmake_minimum_required(VERSION 2.6)

project(libsimd)
enable_testing()

if(IPP8)
    set(IPP_PLATFORM intel64)
else()
    set(IPP_PLATFORM em64t)
    set(MSFX merged${IPP_PLATFORM})
    set(ESFX emerged${IPP_PLATFORM})
    set(PSFX ${IPP_PLATFORM})
endif()

set(LIB_IPP
    ipps${ESFX} ipps${MSFX}
    ippvm${ESFX} ippvm${MSFX}
    ippcore${PSFX}
)

if(CLANG)
    set(CMAKE_C_COMPILER clang)
    set(CMAKE_CXX_COMPILER clang++)
endif()

if(WIN32)
    set(CXX_OPT /EHsc)
else()
    set(CXX_OPT "-std=c++14 -Wextra -Werror -Wno-ignored-attributes -m64 -msse4.1")
    if (AVX)
        set(CXX_OPT "${CXX_OPT} -mavx2")
    endif()
endif()

set(CMAKE_CXX_FLAGS "${CXX_OPT}")
set(CMAKE_EXE_LINKER_FLAGS "${CXX_OPT}")

#add_definitions(-DSEGFAULT_ON_IPP_ERROR)

if(IPP_DIR)
    if(IPP8)
        include_directories(${IPP_DIR}/include)
        link_directories(${IPP_DIR}/lib/${IPP_PLATFORM})
    else()
        include_directories(${IPP_DIR}/${IPP_PLATFORM}/include)
        link_directories(${IPP_DIR}/${IPP_PLATFORM}/lib)
    endif()

    add_library(simdipp STATIC sse_ipp.cpp)
    target_link_libraries(simdipp ${LIB_IPP})
    set_target_properties(simdipp PROPERTIES COMPILE_FLAGS "-DDEPRECATED")
endif()

if(OCL)
    include_directories(/opt/intel/opencl/include)
    link_directories(/opt/intel/opencl)
    add_library(simdocl STATIC ocl.cpp)
    target_link_libraries(simdocl OpenCL)
endif()

add_subdirectory(test)
