find_package(Threads)


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
    
if(Threads_FOUND) # skip bench if pthread is not found

    # Use ExternalProject to get the missig benchmark lib
    if (NOT Benchmark_FOUND)

        # Tell the user we are going to d/l Benchmark
        message(STATUS "Use ExternalProject to get the missig Benchmark lib. It may increase compilation time.")
       
        # Enable ExternalProject CMake module
        INCLUDE(ExternalProject)
    
        # Set default ExternalProject root directory
        SET_DIRECTORY_PROPERTIES(PROPERTIES EP_PREFIX ${CMAKE_BINARY_DIR}/third_party)
    
        ExternalProject_Add(
            goolebenchmark
            GIT_REPOSITORY https://github.com/google/benchmark.git
            INSTALL_COMMAND ""
            CMAKE_ARGS "-DBENCHMARK_ENABLE_GTEST_TESTS=OFF"
            # Wrap download, configure and build steps in a script to log output
            LOG_DOWNLOAD ON
            LOG_CONFIGURE ON
            LOG_BUILD ON
        )
    
        ExternalProject_Get_Property(goolebenchmark source_dir)
        set(Benchmark_INCLUDE_DIRS ${source_dir}/include)
        Include_Directories(${Benchmark_INCLUDE_DIRS})
    
        ExternalProject_Get_Property(goolebenchmark binary_dir)
        set(Benchmark_LIBRARIES ${binary_dir}/src/${CMAKE_FIND_LIBRARY_PREFIXES}benchmark.a)
        set(Benchmark_LIBRARY benchmark)
        add_library(${Benchmark_LIBRARY} UNKNOWN IMPORTED)
        set_target_properties(${Benchmark_LIBRARY} PROPERTIES
            IMPORTED_LOCATION ${Benchmark_LIBRARIES}
        )
    
        add_dependencies(${Benchmark_LIBRARY} goolebenchmark)
        
    endif()
    
    #------------------------------------------------------------------------------------#
    #----------------------------------COMPILATIONS--------------------------------------#
    #------------------------------------------------------------------------------------#
    
    if (USE_CUDA AND NOT USENEWSYNTAX)
    
        CUDA_add_executable(
            bench_grad1conv
            ${CMAKE_CURRENT_SOURCE_DIR}/bench_grad1conv.cu
        )
    
        target_link_libraries(
            bench_grad1conv
            radial_kernels_grad1conv
            benchmark 
        )

    endif()
    
endif() # Threads_FOUND
