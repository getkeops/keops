# Try to find the Google Benchmark library and headers.
#  Benchmark_FOUND        - system has benchmark lib
#  Benchmark_INCLUDE_DIRS - the benchmark include directory
#  Benchmark_LIBRARIES    - libraries needed to use benchmark

find_path(Benchmark_INCLUDE_DIR
    NAMES benchmark/benchmark.h
    NO_SYSTEM_ENVIRONMENT_PATH
    DOC "The directory where benchmark includes reside"
)

find_library(Benchmark_LIBRARY
    NAMES benchmark
    NO_SYSTEM_ENVIRONMENT_PATH
    DOC "The benchmark library"
)

set(Benchmark_INCLUDE_DIRS ${Benchmark_INCLUDE_DIR})
set(Benchmark_LIBRARIES    ${Benchmark_LIBRARY})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Benchmark
    FOUND_VAR Benchmark_FOUND
    REQUIRED_VARS Benchmark_INCLUDE_DIR Benchmark_LIBRARY
)

mark_as_advanced(Benchmark_FOUND)


### - Benchmark
if (Benchmark_FOUND)
    if(${__TYPE__} EQUAL double)
        Set(USE_DOUBLE 1)
    else()
        Set(USE_DOUBLE 0)
    endif()
    
    CUDA_add_executable(
        bench
        ${CMAKE_CURRENT_SOURCE_DIR}/bench.cu
    )
    
    target_link_libraries(
        bench
        shared_obj
        shared_specific
        benchmark 
    )
    
    CUDA_add_library(
        shared_specific SHARED
        "${CMAKE_CURRENT_SOURCE_DIR}/../../convolutions/cuda_grad1conv.cu"
        OPTIONS -D "CUDA_BLOCK_SIZE=192" --pre-include=template.h
    )
    
    set_target_properties( # pass needed options to add_executable()
        bench PROPERTIES
        LINKER_LANGUAGE CXX
    )

else()
    message(STATUS  "   ==>  No bench file can be build.")
endif()
