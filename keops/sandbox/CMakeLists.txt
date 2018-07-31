
# ----------------- test link
#if (USE_CUDA)

    #CUDA_add_executable(
        #test_link_conv
        #${CMAKE_CURRENT_SOURCE_DIR}/test_link.cu
        #OPTIONS --pre-include=formula.h
    #)

    #target_link_libraries(
        #test_link_conv
        #keops
    #)
#endif()

add_executable(
    test_autodiff_cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_autodiff.cpp
)

add_executable(
    test_factorized_cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_factorized.cpp
)

add_executable(
    test_newsyntax_cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_newsyntax.cpp
)

add_executable(
    test_sumgauss_cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_sumgauss.cpp
)

add_executable(
    test_argmin_cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_argmin.cpp
)

add_executable(
    test_argkmin_cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_argkmin.cpp
)

add_executable(
    test_kminargkmin_cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_kminargkmin.cpp
)

add_executable(
    test_minargmin_cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_minargmin.cpp
)

add_executable(
    test_logsumexp_cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_logsumexp.cpp
)

# ----------------- equivalent code to compile_test_factorized
if (USE_CUDA)

    CUDA_add_executable(
        test_autodiff
        ${CMAKE_CURRENT_SOURCE_DIR}/test_autodiff.cu
    )

    CUDA_add_executable(
        test_factorized
        ${CMAKE_CURRENT_SOURCE_DIR}/test_factorized.cu
    )

    CUDA_add_executable(
        test_fromdevice
        ${CMAKE_CURRENT_SOURCE_DIR}/test_fromdevice.cu
    )

    CUDA_add_executable(
        test_sum
        ${CMAKE_CURRENT_SOURCE_DIR}/test_sum.cu
    )

    CUDA_add_executable(
        test_argmin
        ${CMAKE_CURRENT_SOURCE_DIR}/test_argmin.cu
    )

    CUDA_add_executable(
        test_minargmin
        ${CMAKE_CURRENT_SOURCE_DIR}/test_minargmin.cu
    )

    CUDA_add_executable(
        test_logsumexp
        ${CMAKE_CURRENT_SOURCE_DIR}/test_logsumexp.cu
    )

endif()

