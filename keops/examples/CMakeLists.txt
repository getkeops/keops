
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
        test_tensordot_formula_cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/test_tensordot_formula.cpp
)

add_executable(
    test_simple_cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_simple.cpp
)

add_executable(
    test_autodiff_cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_autodiff.cpp
)

add_executable(
    test_reductions_cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_reductions.cpp
)

if (USE_CUDA)

    CUDA_add_executable(
            test_gpytorch
            ${CMAKE_CURRENT_SOURCE_DIR}/test_gpytorch.cu
    )

    CUDA_add_executable(
            test_tensordot_formula
            ${CMAKE_CURRENT_SOURCE_DIR}/test_tensordot_formula.cu
    )

    CUDA_add_executable(
        test_simple
        ${CMAKE_CURRENT_SOURCE_DIR}/test_simple.cu
    )

    CUDA_add_executable(
        test_autodiff
        ${CMAKE_CURRENT_SOURCE_DIR}/test_autodiff.cu
    )

    CUDA_add_executable(
        test_fromdevice
        ${CMAKE_CURRENT_SOURCE_DIR}/test_fromdevice.cu
    )

endif()

