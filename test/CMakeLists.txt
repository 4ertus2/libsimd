include_directories(${PROJECT_SOURCE_DIR})

# nosimd
add_executable(test-common test-common.cpp)
add_executable(test-convert test-convert.cpp)
add_executable(test-arithm test-arithm.cpp)
add_executable(test-arithm-float test-arithm.cpp)
#
set_target_properties(test-common PROPERTIES COMPILE_FLAGS "-DNO_SSE")
set_target_properties(test-convert PROPERTIES COMPILE_FLAGS "-DNO_SSE")
set_target_properties(test-arithm PROPERTIES COMPILE_FLAGS "-DNO_SSE")
set_target_properties(test-arithm-float PROPERTIES COMPILE_FLAGS "-DNO_SSE -DTEST_FLOAT")

# sse
add_executable(test-common-sse test-common.cpp)
add_executable(test-common-sse-unroll test-common.cpp)
add_executable(test-arithm-float-sse test-arithm.cpp)
add_executable(test-arithm-float-sse-unroll test-arithm.cpp)
#
set_target_properties(test-common-sse PROPERTIES COMPILE_FLAGS "-DNO_8_16")
set_target_properties(test-common-sse-unroll PROPERTIES COMPILE_FLAGS "-DNO_8_16 -DUNROLL_MORE")
set_target_properties(test-arithm-float-sse PROPERTIES COMPILE_FLAGS "-DTEST_FLOAT")
set_target_properties(test-arithm-float-sse-unroll PROPERTIES COMPILE_FLAGS "-DTEST_FLOAT -DUNROLL_MORE")
#
#target_link_libraries(test-common-sse simdssef simdssed)
#target_link_libraries(test-common-sse-unroll simdssef_ur simdssed_ur)
#target_link_libraries(test-arithm-float-sse simdssef simdssed)
#target_link_libraries(test-arithm-float-sse-unroll simdssef_ur simdssed_ur)

if(IPP_DIR)
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
target_link_libraries(test-common-ipp simdipp)
target_link_libraries(test-convert-ipp simdipp)
target_link_libraries(test-arithm-ipp simdipp)
target_link_libraries(test-arithm-float-ipp simdipp)
target_link_libraries(test-arithm-fixed-ipp simdipp)
endif(IPP_DIR)

#

add_test(common			test-common)
add_test(common-sse		test-common-sse)
add_test(common-sse-ur	test-common-sse-unroll)
add_test(convert		test-convert)
add_test(arithm			test-arithm)
add_test(arithm_f		test-arithm-float)
add_test(arithm-sse		test-arithm-float-sse)
add_test(arithm-sse-ur	test-arithm-float-sse-unroll)

if(IPP_DIR)
add_test(common_ipp		test-common-ipp)
add_test(convert_ipp	test-convert-ipp)
add_test(arithm_ipp		test-arithm-ipp)
add_test(arithm_f_ipp	test-arithm-float-ipp)
add_test(arithm_fixed	test-arithm-fixed-ipp)
endif()
