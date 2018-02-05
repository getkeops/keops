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

## Options for nvcc
if(CUDA_VERSION_MAJOR LESS 9)
    List(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_20,code=sm_20")
endif()
if(CUDA_VERSION_MAJOR EQUAL 8)
    List(APPEND CUDA_INCLUDE_DIRECTORIES "-Wno-deprecated-gpu-targets")
endif()
List(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_35,code=sm_35")
List(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_60,code=sm_60")
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
