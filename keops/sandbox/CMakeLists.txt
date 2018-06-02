
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
    test_cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test.cpp
)


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

# ----------------- equivalent code to compile_test_factorized
if (USE_CUDA)

    CUDA_add_library(
        simple_autodiff
        ${CMAKE_CURRENT_SOURCE_DIR}/simple_autodiff.cu
    )

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
endif()

