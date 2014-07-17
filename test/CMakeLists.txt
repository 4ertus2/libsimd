include_directories(${PROJECT_SOURCE_DIR})

# nosimd
add_executable(test-common test-common.cpp)
add_executable(test-convert test-convert.cpp)
add_executable(test-arithm test-arithm.cpp)

# unroll
add_executable(test-common-unroll test-common.cpp)
set_target_properties(test-common-unroll PROPERTIES COMPILE_FLAGS -DSIMD_UNROLL)

# ipp
add_executable(test-common-ipp test-common.cpp)
add_executable(test-convert-ipp test-convert.cpp)
add_executable(test-arithm-ipp test-arithm.cpp)
#
set_target_properties(test-common-ipp PROPERTIES COMPILE_FLAGS -DSIMD_IPP)
set_target_properties(test-convert-ipp PROPERTIES COMPILE_FLAGS -DSIMD_IPP)
set_target_properties(test-arithm-ipp PROPERTIES COMPILE_FLAGS -DSIMD_IPP)
#
target_link_libraries(test-common-ipp simd)
target_link_libraries(test-convert-ipp simd)
target_link_libraries(test-arithm-ipp simd)

#

add_test(common			test-common)
add_test(common_unroll	test-common-unroll)
add_test(common_ipp		test-common-ipp)
add_test(convert		test-convert)
add_test(convert_ipp	test-convert-ipp)
add_test(arithm			test-arithm)
add_test(arithm_ipp		test-arithm-ipp)

