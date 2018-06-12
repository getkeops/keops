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
    /home/bcharlier/.local/lib/python3.5/site-packages/torch/lib/include/
    /Users/benoit.martin/miniconda3/pkgs/pytorch-0.4.0-py36_cuda0.0_cudnn0.0_1/lib/python3.6/site-packages/torch/lib/include
    /home/benoit.martin/.conda/pkgs/pytorch-0.4.0-py36_cuda8.0.61_cudnn7.1.2_1/lib/python3.6/site-packages/torch/lib/include
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

if(NOT shared_obj_name)
    SET(shared_obj_name pykeops_module)
endif()

add_definitions(-DMODULE_NAME=${shared_obj_name})

add_subdirectory(pybind11)
pybind11_add_module(${shared_obj_name}
    ${CMAKE_CURRENT_SOURCE_DIR}/common/keops_io.cpp
    )

target_link_libraries(
    ${shared_obj_name} PUBLIC
    keops
)


# set name 
set_target_properties(keops PROPERTIES
    LIBRARY_OUTPUT_NAME  ${shared_obj_name}
    PREFIX ""
)

set_target_properties(${shared_obj_name} PROPERTIES
    LIBRARY_OUTPUT_NAME  ${shared_obj_name}
)

