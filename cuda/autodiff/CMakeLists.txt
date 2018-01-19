## --------------------------------HEADERS-----------------------------------------------
cmake_minimum_required(VERSION 2.8)

project(libkp)

## Find package
find_package(CUDA REQUIRED)
find_package(GTest)
find_package(Matlab)

## Set Path to sources
set(SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}

    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/test
    ${PROJECT_BINARY_DIR}
    )

Include_Directories(${SOURCE_FILES})
#CUDA_INCLUDE_DIRECTORIES(${SOURCE_FILES})


## Options for nvcc
if(CUDA_VERSION_MAJOR LESS 9)
    List(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_20,code=sm_20")
endif()
if(CUDA_VERSION_MAJOR EQUAL 8)
    List(APPEND CUDA_INCLUDE_DIRECTORIES "-Wno-deprecated-gpu-targets")
endif()
List(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_35,code=sm_35")
List(APPEND CUDA_NVCC_FLAGS "--use_fast_math")
List(APPEND CUDA_NVCC_FLAGS "--compiler-options=-fPIC")
List(APPEND CUDA_NVCC_FLAGS "-std=c++11")
List(APPEND CUDA_NVCC_FLAGS "-O2")

## Template macros
add_definitions(-D_FORCE_INLINES)

# - type for computation (float or double)
if(NOT __TYPE__)
    Set(__TYPE__ double)
endif()

## Declare the templates formula. We should generate a file to avoid parsing problem with shell
if(NOT FORMULA)
	Set(FORMULA Scal<Square<Scalprod<_X<1,4>,_Y<3,4>>>,GaussKernel<_P<0>,_X<0,3>,_Y<1,3>,_Y<4,3>>>)
endif()

configure_file(template.h.in template.h @ONLY)
## --------------------------------COMPILATIONS---------------------------------------------

## - create shared lib
CUDA_add_library(
    shared_obj SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/core/link_autodiff.cu
    OPTIONS --pre-include=template.h
)

add_custom_command(
    TARGET shared_obj POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    $<TARGET_FILE:shared_obj>
    \"${FORMULA}\"_${__TYPE__}${CMAKE_SHARED_LIBRARY_SUFFIX}
)

## - mexfiles : generate a working mexfile is painful with cmake. 
##   The problem is twofold : the cuda module do not define a 
##   CUDA_add_object. We then need to use Add_library instead.

if(Matlab_FOUND)

    CUDA_compile( # generate the string "nvcc -c ..." but do not execute it
        mex_file SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/matlab_bindings/cudaconv.cu
        OPTIONS --pre-include=template.h -I${Matlab_INCLUDE_DIRS} -x cu
    )

    add_custom_target(
        mex_cuda ALL
        DEPENDS ${mex_file} # generate the .o as it execute the command made by CUDA_compile
    )

    matlab_get_mex_suffix( # simply get the extension : almost surely mexa64
        ${Matlab_ROOT_DIR} 
        mex_suffix
    )

    add_custom_command( # link the .o by running mex compiler
        TARGET mex_cuda POST_BUILD
        COMMAND ${Matlab_ROOT_DIR}/bin/mex  -L${CUDA_TOOLKIT_ROOT_DIR}/lib64/ -lcudart -L${Matlab_LIBRARIES} -lmex ${mex_file} -output tmp
        COMMAND ${CMAKE_COMMAND} -E rename tmp.${mex_suffix} \"${FORMULA}\"_${__TYPE__}.${mex_suffix}
    )

    set_target_properties( # pass needed options to add_custom_target()
        mex_cuda PROPERTIES 
        PREFIX "" 
        LINKER_LANGUAGE CXX
    )
else()
    message(WARNING "Matlab not found. No mex file produced.")
endif()

## - unit test
if(GTest_FOUND)

    Add_library(
        testic OBJECT 
        ${CMAKE_CURRENT_SOURCE_DIR}/test/unit/gpu_conv.cxx
    )

    Add_executable(
        test_conv
        $<TARGET_OBJECTS:testic>
    )

    target_link_libraries(
        test_conv
        shared_obj
        gtest
        pthread
    )

    set_target_properties( # pass needed options to add_executable()
        test_conv PROPERTIES
        LINKER_LANGUAGE CXX
    )

else()
    message(WARNING "GTest not found. No unit tests build.")
endif()

## - test link
CUDA_compile(
    link_autodiff STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/core/link_autodiff.cu
    OPTIONS --pre-include=template.h
)

CUDA_compile(
    test_link STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/test/test_link.cu
    OPTIONS -D_MWAITXINTRIN_H_INCLUDED
)

CUDA_add_executable(test_link_conv ${test_link} ${link_autodiff})
