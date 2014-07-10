include_directories(${PROJECT_SOURCE_DIR})

add_executable(test-common test-common.cpp)

add_executable(test-common-unroll test-common.cpp)
set_target_properties(test-common-unroll PROPERTIES COMPILE_FLAGS -DSIMD_UNROLL)

add_executable(test-common-ipp test-common.cpp)
set_target_properties(test-common-ipp PROPERTIES COMPILE_FLAGS -DSIMD_IPP)
target_link_libraries(test-common-ipp simd)

#

add_test(common test-common)
add_test(common_unroll test-common-unroll)
add_test(common_ipp test-common-ipp)

