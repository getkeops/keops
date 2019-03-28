#------------------------------------------------------------------------------------#
#------------------------------FIND CUDA AND GPUs------------------------------------#
#------------------------------------------------------------------------------------#

# As of now, we use an ugly mix of old and new cmake methods to properly detect cuda, nvcc and the gpu arch...

# first use cmake to detect cuda, this defines a variable CUDA_FOUND 
find_package(CUDA QUIET)

# set USE_CUDA to CUDA_FOUND, unless USE_CUDA value is enforced
if(NOT DEFINED USE_CUDA)
    Set(USE_CUDA ${CUDA_FOUND})
endif()

# second : we detect Gpu properties with a home-made script
if(CUDA_FOUND AND USE_CUDA)
    # getting some properties of GPUs to pass them as "-D..." options at compilation (adapted from caffe git repo).
    function(caffe_detect_installed_gpus out_variable)
        if(NOT CUDA_gpu_detect_props)
            set(__cufile ${PROJECT_BINARY_DIR}/detect_cuda_props.cu)

            file(WRITE ${__cufile} ""
                "#include <cstdio>\n"
                "int main()\n"
                "{\n"
                "  int count = 0;\n"
                "  if (cudaSuccess != cudaGetDeviceCount(&count)) return -1;\n"
                "  if (count == 0) return -1;\n"
                "  std::printf(\"-DMAXIDGPU=%d;\",count-1);\n"
                "  for (int device = 0; device < count; ++device)\n"
                "  {\n"
                "    cudaDeviceProp prop;\n"
                "    if (cudaSuccess == cudaGetDeviceProperties(&prop, device))\n"
                "      std::printf(\"-DMAXTHREADSPERBLOCK%d=%d;-DSHAREDMEMPERBLOCK%d=%d;\", device, (int)prop.maxThreadsPerBlock, device, (int)prop.sharedMemPerBlock);\n"
                "  }\n"
                "  return 0;\n"
                "}\n")

            execute_process(COMMAND "${CUDA_NVCC_EXECUTABLE}" "--run" "${__cufile}"
                WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/CMakeFiles/"
                RESULT_VARIABLE __nvcc_res OUTPUT_VARIABLE __nvcc_out
                ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

            if(__nvcc_res EQUAL 0)
                set(CUDA_gpu_detect_props ${__nvcc_out} CACHE INTERNAL "Returned GPU architetures from caffe_detect_gpus tool" FORCE)
            endif()
        endif()

        if(NOT CUDA_gpu_detect_props)
            set(${out_variable} FALSE PARENT_SCOPE)
        else()
            set(${out_variable} ${CUDA_gpu_detect_props} PARENT_SCOPE)
        endif()
    endfunction()

    # run the detection 
    if(NOT gpu_compute_props)
        caffe_detect_installed_gpus(gpu_compute_props)
        if(NOT gpu_compute_props)
            set(USE_CUDA FALSE)
            message(STATUS "No GPU detected. USE_CUDA set to FALSE.")
        else()
            message(STATUS "Compute properties automatically set to: ${gpu_compute_props}")
            add_definitions(${gpu_compute_props})
        endif()
    else()
        message(STATUS "Compute properties manually set to ${gpu_compute_props}")
        add_definitions(${gpu_compute_props})
    endif()

endif()


if(CUDA_FOUND AND USE_CUDA)
   
    # Template macros.
    add_definitions(-D_FORCE_INLINES)
    add_definitions(-DCUDA_BLOCK_SIZE=192)

    set(CMAKE_CUDA_COMPILER ${CUDA_NVCC_EXECUTABLE})
    
    enable_language(CUDA)

    if (CMAKE_CUDA_HOST_COMPILER)
        message(STATUS "The CUDA Host CXX Compiler: ${CMAKE_CUDA_HOST_COMPILER}")
    else()
        message(STATUS "The CUDA Host CXX Compiler: ${CMAKE_CXX_COMPILER}")
    endif()


    # Options for nvcc
    CUDA_SELECT_NVCC_ARCH_FLAGS(out_variable "Auto")
    
    set (CUDA_PROPAGATE_HOST_FLAGS ON)
    
    List(APPEND CUDA_NVCC_FLAGS ${out_variable})
    List(APPEND CUDA_NVCC_FLAGS "--use_fast_math")
    List(APPEND CUDA_NVCC_FLAGS "--compiler-options=-fPIC")
    if (CMAKE_CUDA_HOST_COMPILER)
        List(APPEND CUDA_NVCC_FLAGS "-ccbin ${CMAKE_CUDA_HOST_COMPILER}")
    endif()
else()
    set(USE_CUDA 0)
endif()

# this flag is used in pragma
if(USE_CUDA)
    add_definitions(-DUSE_CUDA=1)
else()
    add_definitions(-DUSE_CUDA=0)
endif()

