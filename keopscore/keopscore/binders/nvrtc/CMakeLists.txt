cmake_minimum_required(VERSION 3.12.0)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")

# add_compile_definitions(IGL_STATIC_LIBRARY)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

project(NVRTC_JIT)

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

#################### src ####################
# include
include_directories(
    "${CMAKE_CURRENT_SOURCE_DIR}/"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../"
)

file(GLOB SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/nvrtc_jit.cpp"
)

#################### compile ####################

############################################################
# target library
add_library(nvrtc_jit SHARED ${SOURCES})
target_link_libraries(nvrtc_jit
    ${CUDA_CUDART_LIBRARY}
    ${CUDA_LIBRARIES}
    nvrtc
    cuda
)

