#------------------------------------------------------------------------------------#
#-----------------------------FIND GOOGLE BENCHMARK----------------------------------#
#------------------------------------------------------------------------------------#

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

#------------------------------------------------------------------------------------#
#----------------------------------COMPILATIONS--------------------------------------#
#------------------------------------------------------------------------------------#

if (Benchmark_FOUND AND CUDA_FOUND AND NOT USENEWSYNTAX)

# ----------------- Benchmark

    Set(FORMULA "Grad<GaussKernel<_P<0>,_X<0,3>,_Y<1,3>,_Y<2,3>>,_X<0,3>,_X<3,3>>")
    configure_file(../formula.h.in ${PROJECT_BINARY_DIR}/formula_bench.h @ONLY)

    CUDA_add_library(
        shared_obj_bench SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/../core/link_autodiff.cu
        OPTIONS --pre-include=formula_bench.h
    )

    add_executable(
        bench
        ${CMAKE_CURRENT_SOURCE_DIR}/bench.cpp
    )

    target_link_libraries(
        bench
        shared_obj_bench
        radial_kernels_grad1conv
        benchmark 
    )
    
    set_target_properties( # pass needed options to add_executable()
        bench PROPERTIES
        LINKER_LANGUAGE CXX
    )

else()
    message(STATUS  "   ==> No bench file can be built.")
endif()
