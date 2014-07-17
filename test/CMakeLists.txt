include_directories(${PROJECT_SOURCE_DIR})

# nosimd
add_executable(test-common test-common.cpp)
add_executable(test-convert test-convert.cpp)
add_executable(test-arithm test-arithm.cpp)
add_executable(test-arithm-float test-arithm.cpp)
set_target_properties(test-arithm-float PROPERTIES COMPILE_FLAGS -DTEST_FLOAT)

# unroll
add_executable(test-common-unroll test-common.cpp)
set_target_properties(test-common-unroll PROPERTIES COMPILE_FLAGS -DSIMD_UNROLL)

# ipp
add_executable(test-common-ipp test-common.cpp)
add_executable(test-convert-ipp test-convert.cpp)
add_executable(test-arithm-ipp test-arithm.cpp)
add_executable(test-arithm-float-ipp test-arithm.cpp)
add_executable(test-arithm-fixed-ipp test-arithm.cpp)
#
set_target_properties(test-common-ipp PROPERTIES COMPILE_FLAGS -DSIMD_IPP)
set_target_properties(test-convert-ipp PROPERTIES COMPILE_FLAGS -DSIMD_IPP)
set_target_properties(test-arithm-ipp PROPERTIES COMPILE_FLAGS -DSIMD_IPP)
set_target_properties(test-arithm-float-ipp PROPERTIES COMPILE_FLAGS "-DSIMD_IPP -DTEST_FLOAT")
set_target_properties(test-arithm-fixed-ipp PROPERTIES COMPILE_FLAGS "-DSIMD_IPP -DTEST_FIXED")
#
target_link_libraries(test-common-ipp simd)
target_link_libraries(test-convert-ipp simd)
target_link_libraries(test-arithm-ipp simd)
target_link_libraries(test-arithm-float-ipp simd)
target_link_libraries(test-arithm-fixed-ipp simd)

#

add_test(common			test-common)
add_test(common_unroll	test-common-unroll)
add_test(common_ipp		test-common-ipp)
add_test(convert		test-convert)
add_test(convert_ipp	test-convert-ipp)
add_test(arithm			test-arithm)
add_test(arithm_ipp		test-arithm-ipp)
add_test(arithm_f		test-arithm-float)
add_test(arithm_f_ipp	test-arithm-float-ipp)
add_test(arithm_fixed	test-arithm-fixed-ipp)

