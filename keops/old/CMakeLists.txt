#------------------------------------------------------------------------------------#
#------------------------------------HEADERS-----------------------------------------#
#------------------------------------------------------------------------------------#

cmake_minimum_required(VERSION 3.10)

file(LOCK ${CMAKE_CURRENT_BINARY_DIR} DIRECTORY)
message(STATUS "Lock building directory: ${CMAKE_CURRENT_BINARY_DIR}")

project(KeOps LANGUAGES CXX)

set(KEOPS_SRC ${CMAKE_CURRENT_SOURCE_DIR})

include(${KEOPS_SRC}/cuda.cmake)

## Set Path to sources
set(SOURCE_FILES
    ${KEOPS_SRC}
    ${PROJECT_BINARY_DIR}
    )

Include_Directories(${SOURCE_FILES})

include(headers.cmake)


#------------------------------------------------------------------------------------#
#----------------------------------COMPILATIONS--------------------------------------#
#------------------------------------------------------------------------------------#


# ----------------- create shared lib (cuda)

if(USE_CUDA)

  add_library(
    ${shared_obj_name} SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/core/link_autodiff.cu
    #OPTIONS --pre-include=${shared_obj_name}.h
  )

else()

  # ----------------- create shared lib (cpp)

  add_library(
    ${shared_obj_name} SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/core/link_autodiff.cpp
  )

endif()


target_compile_options(
        ${shared_obj_name} BEFORE
        PRIVATE -include ${shared_obj_name}.h
)

# tell Cmake to explicitly add the dependency: keops is recompiled as soon as formula.h changes.
set_source_files_properties(
        ${CMAKE_CURRENT_SOURCE_DIR}/core/link_autodiff.cpp PROPERTIES
        OBJECT_DEPENDS ${shared_obj_name}.h
)

# shared_lib can be rename if the variable shared_lib_name is provided.
set_target_properties(${shared_obj_name} PROPERTIES
                      LIBRARY_OUTPUT_NAME ${shared_obj_name}
                      PREFIX ""
                      )

# ----------------- various targets

add_subdirectory(specific)
add_subdirectory(sandbox)
add_subdirectory(examples)

if(NOT (${CMAKE_VERSION} VERSION_LESS 3.6.0)) # workaround for https://github.com/google/benchmark/issues/603

  if(NOT (APPLE))
    add_subdirectory(lib/gtest EXCLUDE_FROM_ALL)
    add_subdirectory(test)

    set(BENCHMARK_ENABLE_GTEST_TESTS 0 CACHE INTERNAL "")

    add_subdirectory(lib/benchmark EXCLUDE_FROM_ALL)
    add_subdirectory(bench)
  endif()

endif()

