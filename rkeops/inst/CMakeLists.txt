#------------------------------------------------------------------------------#
#---------------------------------- HEADERS -----------------------------------#
#------------------------------------------------------------------------------#

cmake_minimum_required(VERSION 3.10)

project(rkeops LANGUAGES CXX)
# set up PROJECT_SOURCE_DIR (on which directory cmake is called, 
#   here /path/to/keops/rkeops/include)
# set up PROJECT_BINARY_DIR (where cmake is called, here 
#   /path/to/keops/rkeops/build)

#message(TARGET " PROJECT_SOURCE_DIR=${PROJECT_SOURCE_DIR}")
#message(TARGET " PROJECT_BINARY_DIR=${PROJECT_BINARY_DIR}")

set(SRC_DIR ${PROJECT_SOURCE_DIR}/include)

## C++ API and C++/R binder source
set(KEOPS_SRC_DIR ${SRC_DIR}/keops)
set(BINDER_SRC_DIR ${SRC_DIR}/binder/src)

## build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    set( CMAKE_VERBOSE_MAKEFILE on )
endif()

## include keops header for cuda
include(${KEOPS_SRC_DIR}/cuda.cmake)

## Set Path to sources
set(SOURCE_FILES
    ${SRC_DIR}
    ${KEOPS_SRC_DIR}
    ${BINDER_SRC_DIR}
    ${PROJECT_BINARY_DIR}
)

## include
include_directories(${SOURCE_FILES})
include(${KEOPS_SRC_DIR}/headers.cmake)

## R related include
include_directories(${R_INCLUDE} ${RCPP_INCLUDE} ${RCPPEIGEN_INCLUDE})

## R related dynamic lib
link_directories(${R_LIB})

## installation directory
set(BIN_DIR ${PROJECT_BINARY_DIR/../})

#------------------------------------------------------------------------------#
#-------------------------------- COMPILATIONS --------------------------------#
#------------------------------------------------------------------------------#

################################################################################
##                                   Generic                                  ##
################################################################################

# this dummy flag is used in the bindings
if (${__TYPE__} STREQUAL "double")
    add_definitions(-DUSE_DOUBLE=1)
else()
    add_definitions(-DUSE_DOUBLE=0)
endif()

# ----------------- create shared lib (cuda)
if(USE_CUDA)
  
    add_library(
        keops${shared_obj_name} SHARED
        ${KEOPS_SRC_DIR}/core/link_autodiff.cu
    )

# ----------------- create shared lib (cpp)
else()

    add_library(
        keops${shared_obj_name} SHARED
        ${KEOPS_SRC_DIR}/core/link_autodiff.cpp
    )
endif()


target_compile_options(
    keops${shared_obj_name} BEFORE
    PRIVATE -include ${shared_obj_name}.h
)

# tell Cmake to explicitly add the dependency: keops is recompiled as soon as formula.h change.
set_source_files_properties(
    ${KEOPS_SRC_DIR}/core/link_autodiff.cpp PROPERTIES
    OBJECT_DEPENDS ${shared_obj_name}.h
)

# skip the full RPATH (~run path) for the build tree. We append $ORIGIN later
SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE) 
SET(CMAKE_INSTALL_RPATH "")
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)

# set name
set_target_properties(keops${shared_obj_name} PROPERTIES
    LIBRARY_OUTPUT_NAME ${shared_obj_name}
    PREFIX ""
)

#set_property(TARGET keops${shared_obj_name} PROPERTY CXX_STANDARD 14)

################################################################################
##                                   binding                                  ##
################################################################################

# ----------------- keops related files
add_library( # generate the string "g++ -c ..." but do not execute it
    ${shared_obj_name} SHARED
    ${BINDER_SRC_DIR}/generic_red.cpp
)

# target
target_compile_options(
    ${shared_obj_name} BEFORE
    PRIVATE -include ${shared_obj_name}.h
)

#set_property(TARGET ${shared_obj_name} PROPERTY CXX_STANDARD 11)

# Ensure the shared lib look for the other .so in its own dir.
if(APPLE)
    set_target_properties(
        ${shared_obj_name} 
        PROPERTIES LINK_FLAGS 
        "-Wl,-rpath,@loader_path/."
    )
else()
    set_target_properties(
        ${shared_obj_name} 
        PROPERTIES LINK_FLAGS "-Wl,-rpath,$ORIGIN"
    )
endif()

# target
target_link_libraries(
    ${shared_obj_name} PUBLIC
    keops${shared_obj_name}
)


# ----------------- Rcpp/RcppEigen related files
add_library( # generate the string "g++ -c ..." but do not execute it
    rkeops${shared_obj_name} SHARED
    # binder source
    ${BINDER_SRC_DIR}/r_genred.cpp
    ${BINDER_SRC_DIR}/RcppExports.cpp
    ${BINDER_SRC_DIR}/test_binder.cpp
)

#set_property(TARGET ${shared_obj_name} PROPERTY CXX_STANDARD 11)

# Ensure the shared lib look for the other .so in its own dir.
if(APPLE)
    set_target_properties(
        rkeops${shared_obj_name} 
        PROPERTIES LINK_FLAGS 
        "-Wno-ignored-attributes -Wl,-rpath,@loader_path/. -lR -Wl,-framework -Wl,CoreFoundation"
    )
else()
    set_target_properties(
        rkeops${shared_obj_name} 
        PROPERTIES LINK_FLAGS "-Wno-ignored-attributes -Wl,-rpath,$ORIGIN"
    )
endif()

target_compile_options(
    rkeops${shared_obj_name} 
    PRIVATE -Wno-ignored-attributes)

# target
target_link_libraries(
    rkeops${shared_obj_name} PUBLIC
    ${shared_obj_name}
)

## Installation step
#install(
#    TARGETS ${shared_obj_name}
#    LIBRARY DESTINATION ${PROJECT_BINARY_DIR}
#    PUBLIC_HEADER DESTINATION ${PROJECT_BINARY_DIR}
#)
#message(-------------- $<TARGET_FILE:${shared_obj_name}>)
#add_custom_command(
#    TARGET ${shared_obj_name} POST_BUILD
#    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${shared_obj_name}> ${BIN_DIR}
#)
#add_custom_command(
#    TARGET keops${shared_obj_name} POST_BUILD
#    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:keops${shared_obj_name}> ${BIN_DIR}
#)


## Write a log file to decypher keops dllname
#if(commandLine)
#   string(TIMESTAMP TODAY "%Y/%m/%d")
#   file(APPEND ${PROJECT_BINARY_DIR}/keops_hash.log
#     "# ${shared_obj_name} compiled on ${TODAY} with ${CMAKE_CXX_COMPILER} (${CMAKE_CXX_COMPILER_VERSION}):\n\n ${commandLine}\n cmake --build . --target ${shared_obj_name} --  VERBOSE=1\n\n# ----------------------------------------------------------------------\n")
#endif()
