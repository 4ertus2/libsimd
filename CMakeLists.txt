cmake_minimum_required(VERSION 2.6)

project(libsimd)
enable_testing()

set(SFX em64t)
set(MSFX merged${SFX})
set(ESFX emerged${SFX})

set(LIB_IPP
	ippsr${ESFX} ippsr${MSFX}
	ipps${ESFX} ipps${MSFX}
	ippvm${ESFX} ippvm${MSFX}
	ippcore${SFX}
)

if(WIN32)
	set(CXX_OPT /EHsc)
	set(CMAKE_CXX_FLAGS "${CXX_OPT}")
	set(CMAKE_EXE_LINKER_FLAGS "${CXX_OPT}")
endif()

#add_definitions(-DSEGFAULT_ON_IPP_ERROR)

include_directories(${IPP_INC_DIR})
link_directories(${IPP_LIB_DIR})

add_library(simd STATIC sse_ipp.cpp)
target_link_libraries(simd ${LIB_IPP})

add_subdirectory(test)
