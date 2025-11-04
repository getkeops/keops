cmake_minimum_required(VERSION 3.10)
project(***project_name***)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)

# Set compiler options
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /LD /Ox /GL /openmp")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -O3 -fopenmp")
endif()

# Preprocessor macros
***macros***

# Include directories
***includes***

# Link directories
***link_dirs***

# Source file
set(SOURCE_FILE "***source_file***")

# Add shared library
add_library(${PROJECT_NAME} SHARED ${SOURCE_FILE})

# Link
***links***

# Set the output path for the shared library and ensure the .pyd extension
set_target_properties(${PROJECT_NAME} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    OUTPUT_NAME "***project_name***"
    SUFFIX "***suffix***"  # Ensure the correct file extension for Python modules
    MSVC_WARNING_LEVEL 0
    MSVC_WARNING_DISABLE 8029 # Warning because we are building in a tmp dir
)
