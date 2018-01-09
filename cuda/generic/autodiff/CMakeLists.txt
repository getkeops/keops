## --------------------------------HEADERS-----------------------------------------------

cmake_minimum_required(VERSION 2.8)

## Find package
find_package(CUDA REQUIRED)
find_package(GTest REQUIRED)

## Set Path to sources
set(SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}

    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/test
    ${PROJECT_BINARY_DIR}
    )

CUDA_INCLUDE_DIRECTORIES(${SOURCE_FILES})


## Options for nvcc
List(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_35,code=sm_35")
List(APPEND CUDA_NVCC_FLAGS "--use_fast_math")
List(APPEND CUDA_NVCC_FLAGS "--compiler-options=-fPIC")
List(APPEND CUDA_NVCC_FLAGS "-std=c++11")
List(APPEND CUDA_NVCC_FLAGS "-O2")


add_definitions(-D_FORCE_INLINES)
add_definitions(-D__TYPE__ float)

## --------------------------------COMPILATIONS---------------------------------------------

## Declare the templates formula. We should generate a file to avoid parsing problem with shell
if(NOT FORMULA)
	Set(FORMULA Scal<Square<Scalprod<_X<1,4>,_Y<3,4>>>,GaussKernel<_P<0>,_X<0,3>,_Y<1,3>,_Y<4,3>>>)
endif()
configure_file(formula.h.in formula.h @ONLY)


CUDA_compile(
	link_autodiff STATIC
        formula.h
	${CMAKE_CURRENT_SOURCE_DIR}/core/link_autodiff.cu
)

CUDA_compile(
	test_link STATIC
	${CMAKE_CURRENT_SOURCE_DIR}/test/test_link.cu
    OPTIONS -D_MWAITXINTRIN_H_INCLUDED
)

CUDA_add_executable(test_conv ${test_link} ${link_autodiff})


## ------------------------------------------------------------------------------------------
