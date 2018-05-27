#------------------------------------------------------------------------------------#
#------------------------------------HEADERS-----------------------------------------#
#------------------------------------------------------------------------------------#

cmake_minimum_required(VERSION 2.8)

project(KeOps LANGUAGES CXX)

## Set Path to sources
set(SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_BINARY_DIR}
)

Include_Directories(${SOURCE_FILES})

include(headers.cmake)

#------------------------------------------------------------------------------------#
#----------------------------------COMPILATIONS--------------------------------------#
#------------------------------------------------------------------------------------#

    
# ----------------- create shared lib (cuda)

if(USE_CUDA)
  
    CUDA_add_library(
        keops SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/core/link_autodiff.cu
        OPTIONS --pre-include=formula.h
    )

else()

# ----------------- create shared lib (cpp)

    add_library(
        keops SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/core/link_autodiff.cpp
    )

    target_compile_options(
        keops BEFORE
        PRIVATE -include formula.h
    )

    # tell Cmake to explicitly add the dependency: keops is recompiled as soon as formula.h changes.
    set_source_files_properties(
        ${CMAKE_CURRENT_SOURCE_DIR}/core/link_autodiff.cpp PROPERTIES
        OBJECT_DEPENDS formula.h
    )

endif()

# shared_lib can be rename if the variable shared_lib_name is provided.
if(shared_obj_name)

    set_target_properties(keops PROPERTIES
        LIBRARY_OUTPUT_NAME  ${shared_obj_name}
        PREFIX ""
    )

endif()

# ----------------- various targets

add_subdirectory(specific)

if(NOT PYTHON_LIB)
    if ((NOT APPLE) AND (NOT(${CMAKE_VERSION} VERSION_LESS 3.6.0))) # workaround for https://github.com/google/benchmark/issues/603
        add_subdirectory(bench)
        add_subdirectory(test)
    endif()
        add_subdirectory(sandbox)
endif()
