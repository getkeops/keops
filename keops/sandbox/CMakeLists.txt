
# ----------------- test link
if (USE_CUDA)

    CUDA_add_executable(
        test_link_conv
        ${CMAKE_CURRENT_SOURCE_DIR}/test_link.cu
        OPTIONS --pre-include=formula.h
    )

    target_link_libraries(
        test_link_conv
        keops
    )
endif()

# ----------------- equivalent code to compile_test_autodiff_cpp

add_executable(
    test_autodiff_cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_autodiff.cpp
)


# ----------------- equivalent code to compile_test_factorized
if (USE_CUDA)

    CUDA_add_executable(
        test_factorized
        ${CMAKE_CURRENT_SOURCE_DIR}/test_factorized.cu
    )

endif()

# ----------------- equivalent code to compile_test_factorized_cpp
add_executable(
    test_factorized_cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_factorized.cpp
    )
