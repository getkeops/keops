#------------------------------------------------------------------------------------#
#------------------------------------HEADERS-----------------------------------------#
#------------------------------------------------------------------------------------#

cmake_minimum_required(VERSION 2.8)

project(PyKeOps LANGUAGES CXX)

## Set Path to sources
set(PROJECT_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../keops)

set(SOURCE_FILES
    ${PROJECT_SRC}
    ${PROJECT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/pybind11
)

Include_Directories(${SOURCE_FILES})

include(${PROJECT_SRC}/headers.cmake)


#------------------------------------------------------------------------------------#
#----------------------------------COMPILATIONS--------------------------------------#
#------------------------------------------------------------------------------------#

# ----------------- create shared lib (cuda)

if(USE_CUDA)
  
    CUDA_add_library(
        keops SHARED
        ${PROJECT_SRC}/core/link_autodiff.cu
        OPTIONS --pre-include=formula.h
    )

else()

# ----------------- create shared lib (cpp)

    add_library(
        keops SHARED
        ${PROJECT_SRC}/core/link_autodiff.cpp
    )

    target_compile_options(
        keops BEFORE
        PRIVATE -include formula.h
    )

    # tell Cmake to explicitly add the dependency: keops is recompiled as soon as formula.h changes.
    set_source_files_properties(
        ${PROJECT_SRC}/core/link_autodiff.cpp PROPERTIES
        OBJECT_DEPENDS formula.h
    )

endif()


add_subdirectory(pybind11)
pybind11_add_module(pykeops_module
    ${CMAKE_CURRENT_SOURCE_DIR}/common/keops_io.cpp
    )

target_link_libraries(
    pykeops_module PUBLIC
    keops
)


# shared_lib can be rename if the variable shared_lib_name is provided.
if(shared_obj_name)

    set_target_properties(keops PROPERTIES
        LIBRARY_OUTPUT_NAME  ${shared_obj_name}
        PREFIX ""
    )

    set_target_properties(pykeops_module PROPERTIES
        LIBRARY_OUTPUT_NAME  ${shared_obj_name}
    )

endif()
