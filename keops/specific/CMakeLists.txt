if (USE_CUDA)

    CUDA_add_library(
        radial_kernels_conv SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/radial_kernels/cuda_conv.cu
    )

    CUDA_add_library(
        radial_kernels_grad1conv SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/radial_kernels/cuda_grad1conv.cu
    )

    #CUDA_add_library(
        #radial_kernels_gradconv_xa SHARED
        #${CMAKE_CURRENT_SOURCE_DIR}/radial_kernels/cuda_gradconv_xa.cu
    #)

    #CUDA_add_library(
        #radial_kernels_gradconv_xb SHARED
        #${CMAKE_CURRENT_SOURCE_DIR}/radial_kernels/cuda_gradconv_xb.cu
    #)

    #CUDA_add_library(
        #radial_kernels_gradconv_xx SHARED
        #${CMAKE_CURRENT_SOURCE_DIR}/radial_kernels/cuda_gradconv_xx.cu
    #)

    #CUDA_add_library(
        #radial_kernels_gradconv_xy SHARED
        #${CMAKE_CURRENT_SOURCE_DIR}/radial_kernels/cuda_gradconv_xy.cu
    #)

endif()
