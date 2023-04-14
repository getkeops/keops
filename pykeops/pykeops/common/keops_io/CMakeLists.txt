# example
# cmake -G "Visual Studio 16 2019" -A x64 ../ -Dpybind11_DIR=C:\ProgramData\Anaconda3\envs\pt1.11_cu11.3\ lib\site-packages\pybind11\share\cmake\pybind11

cmake_minimum_required(VERSION 3.12.0)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")

# add_compile_definitions(IGL_STATIC_LIBRARY)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

project(PYKEOPS_NVRTC)

#################### macro ####################
add_definitions(-DARCHTAG=\"sm\")
add_definitions(-DnvrtcGetTARGET=nvrtcGetCUBIN)
add_definitions(-DnvrtcGetTARGETSize=nvrtcGetCUBINSize)

#################### dependency ####################
# cuda
find_package(CUDA REQUIRED)
set(CUDA_ARCH_BIN "86" CACHE STRING "Specify 'real' GPU arch to build binaries for, BIN(PTX) format is supported. Example: 1.3 2.1(1.3) or 13 21(13)")
set(CUDA_ARCH_PTX "86" CACHE STRING "Specify 'virtual' PTX arch to build PTX intermediate code for. Example: 1.0 1.2 or 10 12")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -ccbin g++ -O3) # -std=c++11  -Xptxas -g -allow-expensive-optimizations=true

include_directories(${CUDA_INCLUDE_DIRS})
message("CUDA_CUDART_LIBRARY = ${CUDA_CUDART_LIBRARY}")
message("CUDA_LIBRARIES = ${CUDA_LIBRARIES}")
message("CUDA_TOOLKIT_ROOT_DIR = ${CUDA_TOOLKIT_ROOT_DIR}")
message("set nvrtc path = ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64")
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)

# pybind11
find_package(pybind11 REQUIRED)
include_directories(${pybind11_INCLUDE_DIRS})
message("pybind11_INCLUDE_DIRS = ${pybind11_INCLUDE_DIRS}")
message("PYTHON_LIBRARY = ${PYTHON_LIBRARY}")

#################### src ####################
# include
include_directories(
    "${CMAKE_CURRENT_SOURCE_DIR}/"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../keopscore/keopscore/"
)

file(GLOB SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/pykeops_nvrtc.cpp"
)

#################### compile ####################

############################################################
# target library
add_library(pykeops_nvrtc SHARED ${SOURCES})
target_link_libraries(pykeops_nvrtc
    ${CUDA_CUDART_LIBRARY}
    ${CUDA_LIBRARIES}
    nvrtc
    cuda
    ${PYTHON_LIBRARY}
)

# change .dll to .pyd
SET_TARGET_PROPERTIES( pykeops_nvrtc
   PROPERTIES
   SUFFIX ".pyd"
)

