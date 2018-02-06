## --------------------------------HEADERS-----------------------------------------------
cmake_minimum_required(VERSION 2.8)

project(libkp LANGUAGES CXX)

## Find package
find_package(CUDA)

## Set Path to sources
set(SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}

    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/core/formulas
    ${CMAKE_CURRENT_SOURCE_DIR}/core/reductions
    ${CMAKE_CURRENT_SOURCE_DIR}/test
    ${CMAKE_CURRENT_SOURCE_DIR}/test/unit
    ${PROJECT_BINARY_DIR}
)

Include_Directories(${SOURCE_FILES})

set (CMAKE_CXX_FLAGS "--std=c++11 -O3")

set (CUDA_PROPAGATE_HOST_FLAGS ON)






#--------------------------------find GPUs------------------------------------
# A function for automatic detection of GPUs installed  (if autodetection is enabled)
# Usage:
#   caffe_detect_installed_gpus(out_variable)
# source: caffe git repo.

function(caffe_detect_installed_gpus out_variable)
    if(NOT CUDA_gpu_detect_output)
        set(__cufile ${PROJECT_BINARY_DIR}/detect_cuda_archs.cu)

        file(WRITE ${__cufile} ""
            "#include <cstdio>\n"
            "int main()\n"
            "{\n"
            "  int count = 0;\n"
            "  if (cudaSuccess != cudaGetDeviceCount(&count)) return -1;\n"
            "  if (count == 0) return -1;\n"
            "  for (int device = 0; device < count; ++device)\n"
            "  {\n"
            "    cudaDeviceProp prop;\n"
            "    if (cudaSuccess == cudaGetDeviceProperties(&prop, device))\n"
            "      std::printf(\"%d.%d \", prop.major, prop.minor);\n"
            "  }\n"
            "  return 0;\n"
            "}\n")

        execute_process(COMMAND "${CUDA_NVCC_EXECUTABLE}" "--run" "${__cufile}"
            WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/CMakeFiles/"
            RESULT_VARIABLE __nvcc_res OUTPUT_VARIABLE __nvcc_out
            ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

        if(__nvcc_res EQUAL 0)
            string(REPLACE "2.1" "2.1(2.0)" __nvcc_out "${__nvcc_out}")
            set(CUDA_gpu_detect_output ${__nvcc_out} CACHE INTERNAL "Returned GPU architetures from caffe_detect_gpus tool" FORCE)
        endif()
    endif()

    if(NOT CUDA_gpu_detect_output)
        message(STATUS "Automatic GPU detection failed. Building for all known architectures.")
        set(${out_variable} ${Caffe_known_gpu_archs} PARENT_SCOPE)
    else()
        set(${out_variable} ${CUDA_gpu_detect_output} PARENT_SCOPE)
    endif()
endfunction()

# run the detection 
if(NOT gpu_compute_capability)
    caffe_detect_installed_gpus(gpu_compute_capability)
    message("-- Found GPU: compute capability ${gpu_compute_capability}")
    # remove dots and convert to lists
    string(REGEX REPLACE "\\." "" __cuda_arch_bin "${gpu_compute_capability}")
endif()





## Options for nvcc
if(CUDA_VERSION_MAJOR EQUAL 8)
    List(APPEND CUDA_INCLUDE_DIRECTORIES "-Wno-deprecated-gpu-targets")
endif()
List(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_${gpu_compute_capability},code=sm_${gpu_compute_capability}")
List(APPEND CUDA_NVCC_FLAGS "--use_fast_math")
List(APPEND CUDA_NVCC_FLAGS "--compiler-options=-fPIC")

## Template macros
add_definitions(-D_FORCE_INLINES)

# - type for computation (float or double)
if(NOT __TYPE__)
    Set(__TYPE__ float)
endif()

# - Declare the templates formula. We should generate a file to avoid parsing problem with shell
if(NOT FORMULA)
    Set(FORMULA Grad<GaussKernel<_P<0>,_X<0,3>,_Y<0,3>,_Y<1,3>>,_X<0,3>,_X<1,3>>)
endif()

if(NOT USENEWSYNTAX)
    set(USENEWSYNTAX FALSE)
endif()

configure_file(template.h.in template.h @ONLY)

## --------------------------------COMPILATIONS---------------------------------------------
    
## - create shared lib

if(CUDA_FOUND)
    CUDA_add_library(
        shared_obj SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/core/link_autodiff.cu
        OPTIONS --pre-include=template.h
    )
endif()

add_library(
    shared_obj_cpp SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/core/link_autodiff.cpp
)

target_compile_options(
    shared_obj_cpp BEFORE
    PRIVATE -include template.h
)


# shared_lib can be rename if the variable share_lib_name is provided.
if(shared_obj_name)
    #Set(shared_lib_name \"${FORMULA}\"_${__TYPE__}${CMAKE_SHARED_LIBRARY_SUFFIX})
    add_custom_command(
        TARGET shared_obj POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E rename
        $<TARGET_FILE:shared_obj>
        ${shared_obj_name}    
    )
endif()

### - test link

CUDA_add_executable(test_link_conv #${test_link})
    ${CMAKE_CURRENT_SOURCE_DIR}/test/test_link.cu
    OPTIONS -D_MWAITXINTRIN_H_INCLUDED
)

target_link_libraries(
    test_link_conv
    shared_obj
)


add_subdirectory(bench)
add_subdirectory(test)
add_subdirectory(matlab_bindings)
