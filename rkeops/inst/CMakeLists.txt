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

## include keops header for cuda
include(${KEOPS_SRC_DIR}/cuda.cmake)

## Set Path to sources
set(SOURCE_FILES
    ${KEOPS_SRC_DIR}
    ${BINDER_SRC_DIR}
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

# ----------------- binding
add_library( # generate the string "g++ -c ..." but do not execute it
    rkeops${shared_obj_name} SHARED
    # binder source
    ${BINDER_SRC_DIR}/generic-generic_red.cpp
    ${BINDER_SRC_DIR}/RcppExports.cpp
    ${BINDER_SRC_DIR}/specific-convolutions-cudagrad1conv.cpp
    ${BINDER_SRC_DIR}/specific-convolutions-radial_kernel_conv.cpp
    ${BINDER_SRC_DIR}/specific-shape_distance-cudafshape.cpp
    ${BINDER_SRC_DIR}/specific-shape_distance-cudafshape_df.cpp
    ${BINDER_SRC_DIR}/specific-shape_distance-cudafshape_dx.cpp
    ${BINDER_SRC_DIR}/specific-shape_distance-cudafshape_dxi.cpp
    ${BINDER_SRC_DIR}/test_binder.cpp
)

install(
    TARGETS rkeops${shared_obj_name}
    LIBRARY DESTINATION ${PROJECT_BINARY_DIR}
    PUBLIC_HEADER DESTINATION ${PROJECT_BINARY_DIR}
)

#################################################################################
##                                   Generic                                    #
#################################################################################

## this dummy flag is used in the bindings
#if (${__TYPE__} STREQUAL "double")
#    add_definitions(-DUSE_DOUBLE=1)
#else()
#    add_definitions(-DUSE_DOUBLE=0)
#endif()

## ----------------- create shared lib (cuda)
#if(USE_CUDA)
#  
#    CUDA_add_library(
#        keops${shared_obj_name} SHARED
#        ${KEOPS_SRC_DIR}/core/link_autodiff.cu
#        OPTIONS --pre-include=${shared_obj_name}.h
#    )
#
## ----------------- create shared lib (cpp)
#else()
#
#    add_library(
#        keops${shared_obj_name} SHARED
#        ${KEOPS_SRC_DIR}/core/link_autodiff.cpp
#    )
#
#    target_compile_options(
#        keops${shared_obj_name} BEFORE
#        PRIVATE -include ${shared_obj_name}.h
#    )
#
#    # tell Cmake to explicitly add the dependency: 
#    # keops is recompiled as soon as formula.h change.
#    set_source_files_properties(
#        ${KEOPS_SRC_DIR}/core/link_autodiff.cpp PROPERTIES
#        OBJECT_DEPENDS ${shared_obj_name}.h
#    )
#
#endif()

## skip the full RPATH (~run path) for the build tree. We append $ORIGIN later
#SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
#SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE) 
#SET(CMAKE_INSTALL_RPATH "")
#SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)
#
## set name
#set_target_properties(keops${shared_obj_name} PROPERTIES
#    LIBRARY_OUTPUT_NAME  ${shared_obj_name}
#    PREFIX ""
#)
#
## ----------------- binding
#add_library( # generate the string "g++ -c ..." but do not execute it
#    rkeops${shared_obj_name} SHARED
#    # binder source
#    ${BINDER_SRC_DIR}/generic-generic_red.cpp
#    ${BINDER_SRC_DIR}/RcppExports.cpp
#    ${BINDER_SRC_DIR}/specific-convolutions-cudagrad1conv.cpp
#    ${BINDER_SRC_DIR}/specific-convolutions-radial_kernel_conv.cpp
#    ${BINDER_SRC_DIR}/specific-shape_distance-cudafshape.cpp
#    ${BINDER_SRC_DIR}/specific-shape_distance-cudafshape_df.cpp
#    ${BINDER_SRC_DIR}/specific-shape_distance-cudafshape_dx.cpp
#    ${BINDER_SRC_DIR}/specific-shape_distance-cudafshape_dxi.cpp
#    ${BINDER_SRC_DIR}/test_binder.cpp
#)
#
## target
#target_compile_options(
#    rkeops${shared_obj_name} BEFORE
#    PRIVATE -include ${shared_obj_name}.h
#)
#
## Ensure the shared lib look for the other .so in its own dir.
#if(APPLE)
#    set_target_properties(
#        rkeops${shared_obj_name} 
#        PROPERTIES LINK_FLAGS 
#        "-Wl,-rpath,@loader_path/."
#    )
#else()
#    set_target_properties(
#        rkeops${shared_obj_name} 
#        PROPERTIES LINK_FLAGS "-Wl,-rpath,$ORIGIN"
#    )
#endif()
#
## target
#target_link_libraries(
#    rkeops${shared_obj_name} PUBLIC
#    keops${shared_obj_name}
#)
#
## Installation step
#install(
#    TARGETS rkeops${shared_obj_name}
#    LIBRARY DESTINATION ${PROJECT_BINARY_DIR}
#    PUBLIC_HEADER DESTINATION ${PROJECT_BINARY_DIR}
#)
#add_custom_command(
#    TARGET keops${shared_obj_name} POST_BUILD
#    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:keops${shared_obj_name}> ${BIN_DIR}
#)
#add_custom_command(
#    TARGET rkeops${shared_obj_name} POST_BUILD
#    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:keops${shared_obj_name}> ${BIN_DIR}
#)
#
#
##
### Write a log file to decypher keops dllname
##if(commandLine)
##   string(TIMESTAMP TODAY "%Y/%m/%d")
##   file(APPEND ${PROJECT_BINARY_DIR}/keops_hash.log
##     "# ${shared_obj_name} compiled on ${TODAY} with ${CMAKE_CXX_COMPILER} (${CMAKE_CXX_COMPILER_VERSION}):\n\n ${commandLine}\n cmake --build . --target ${shared_obj_name} --  VERBOSE=1\n\n# ----------------------------------------------------------------------\n")
##endif()
##
##
##
##
###################################################################################
####                                   Specific                                   #
###################################################################################
###
###if (USE_CUDA)
###
####------------------------------------ CONV ------------------------------------#
###    CUDA_add_library(
###        radial_kernel_conv_cuda SHARED
###        ${KEOPS_SRC_DIR}/specific/radial_kernels/cuda_conv.cu
###    )
###
###    #pybind11_add_module(radial_kernel_conv
###    #    ${CMAKE_CURRENT_SOURCE_DIR}/numpy/convolutions/radial_kernel_conv.cpp
###    #)
###
###    target_compile_options(
###        radial_kernel_conv BEFORE
###        PRIVATE  -include ${shared_obj_name}.h
###    )
###
###    target_link_libraries(
###        radial_kernel_conv PUBLIC
###        radial_kernel_conv_cuda
###    )
###
###    # Ensure the shared lib look for the other .so in its own dir.
###    if(APPLE)
###        set_target_properties(radial_kernel_conv PROPERTIES LINK_FLAGS "-Wl,-rpath,@loader_path/.")
###    else()
###        set_target_properties(radial_kernel_conv PROPERTIES LINK_FLAGS "-Wl,-rpath,$ORIGIN")
###    endif()
###
###    # Installation step
###    add_custom_command(
###        TARGET radial_kernel_conv POST_BUILD
###        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:radial_kernel_conv> ${BIN_DIR}
###    )
###    add_custom_command(
###        TARGET radial_kernel_conv_cuda POST_BUILD
###        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:radial_kernel_conv_cuda> ${BIN_DIR}
###    )
###
###
####--------------------------------- GRAD1CONV ----------------------------------#
###
###    CUDA_add_library(
###        radial_kernel_grad1conv_cuda SHARED
###        ${KEOPS_SRC_DIR}/specific/radial_kernels/cuda_grad1conv.cu
###    )
###
###    pybind11_add_module(radial_kernel_grad1conv
###        ${CMAKE_CURRENT_SOURCE_DIR}/numpy/convolutions/radial_kernel_grad1conv.cpp
###    )
###
###    target_compile_options(
###        radial_kernel_grad1conv BEFORE
###        PRIVATE  -include ${shared_obj_name}.h
###    )
###
###    target_link_libraries(
###        radial_kernel_grad1conv PUBLIC
###        radial_kernel_grad1conv_cuda
###    )
###
###    # Ensure the shared lib look for the other .so in its own dir.
###    if(APPLE)
###        set_target_properties(radial_kernel_grad1conv PROPERTIES LINK_FLAGS "-Wl,-rpath,@loader_path/.")
###    else()
###        set_target_properties(radial_kernel_grad1conv PROPERTIES LINK_FLAGS "-Wl,-rpath,$ORIGIN")
###    endif()
###
###    # Installation step
###    add_custom_command(
###        TARGET radial_kernel_grad1conv POST_BUILD
###        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:radial_kernel_grad1conv> ${BIN_DIR}
###    )
###    add_custom_command(
###        TARGET radial_kernel_grad1conv_cuda POST_BUILD
###        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:radial_kernel_grad1conv_cuda> ${BIN_DIR}
###    )
###
###
####--------------------------------- FSHAPE SCP ---------------------------------#
###
###    if(NOT KERNEL_GEOM OR (KERNEL_GEOM STREQUAL "gaussian"))
###        SET(KERNEL_GEOM "gaussian")
###        SET(KERNEL_GEOM_TYPE 0)
###    elseif(KERNEL_GEOM STREQUAL "cauchy")
###            SET(KERNEL_GEOM_TYPE 1)
###    else()
###        message(FATAL_ERROR "Set KERNEL_GEOM type to gaussian or cauchy.")
###    endif()
###    add_definitions(-DKERNEL_GEOM_TYPE=${KERNEL_GEOM_TYPE})
###
###    if(NOT KERNEL_SIG OR (KERNEL_SIG STREQUAL gaussian))
###        SET(KERNEL_SIG "gaussian")
###        SET(KERNEL_SIG_TYPE 0)
###    elseif(KERNEL_SIG STREQUAL cauchy)
###        SET(KERNEL_SIG_TYPE 1)
###    else()
###        message(FATAL_ERROR "Set KERNEL_SIG type to gaussian or cauchy.")
###    endif()
###    add_definitions(-DKERNEL_SIG_TYPE=${KERNEL_SIG_TYPE})
###
###    if(NOT KERNEL_SPHERE OR (KERNEL_SPHERE STREQUAL gaussian_unoriented))
###        SET(KERNEL_SPHERE "gaussian_unoriented")
###        SET(KERNEL_SPHERE_TYPE 0)
###    elseif(KERNEL_SPHERE STREQUAL binet)
###        SET(KERNEL_SPHERE_TYPE 1)
###    elseif(KERNEL_SPHERE STREQUAL gaussian_oriented)
###        SET(KERNEL_SPHERE_TYPE 2)
###    elseif(KERNEL_SPHERE STREQUAL linear)
###        SET(KERNEL_SPHERE_TYPE 3)
###    else()
###        message(FATAL_ERROR "Set KERNEL_SPHERE type to gaussian_unoriented, binet, gaussian_oriented or linear.")
###    endif()
###    add_definitions(-DKERNEL_SPHERE_TYPE=${KERNEL_SPHERE_TYPE})
###
###    #foreach(ext_name "" "_dx" "_df" "_dxi")
###    foreach(ext_name "")
###
###        SET(fshape_scp_name fshape_scp${ext_name}_${KERNEL_GEOM}${KERNEL_SIG}${KERNEL_SPHERE}_${__TYPE__})
###
###        SET(name1 fshape_gpu${ext_name})
###
###        CUDA_add_library(
###            ${name1} SHARED
###            ${KEOPS_SRC_DIR}/specific/shape_distance/${name1}.cu
###        )
###
###        set_target_properties(${name1} PROPERTIES
###            LIBRARY_OUTPUT_NAME ${fshape_scp_name}
###            PREFIX ""
###        )
###
###        SET(name2 fshape_scp${ext_name})
###        add_definitions(-DMODULE_NAME_FSHAPE_SCP=${fshape_scp_name})
###
###        pybind11_add_module(${fshape_scp_name}
###            ${CMAKE_CURRENT_SOURCE_DIR}/numpy/shape_distance/${name2}.cpp
###        )
###
###        target_compile_options(
###            ${fshape_scp_name} BEFORE
###            PRIVATE  -include ${shared_obj_name}.h
###        )
###
###        target_link_libraries(
###            ${fshape_scp_name} PUBLIC
###            ${name1}
###        )
###        # Ensure the shared lib look for the other .so in its own dir.
###        if(APPLE)
###            set_target_properties(${fshape_scp_name} PROPERTIES LINK_FLAGS "-Wl,-rpath,@loader_path/.")
###        else()
###            set_target_properties(${fshape_scp_name} PROPERTIES LINK_FLAGS "-Wl,-rpath,$ORIGIN")
###        endif()
###
###        # Installation step
###        add_custom_command(
###            TARGET ${fshape_scp_name} POST_BUILD
###            COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${fshape_scp_name}> ${BIN_DIR}
###        )
###        add_custom_command(
###            TARGET ${name1} POST_BUILD
###            COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${name1}> ${BIN_DIR}
###        )
###
###
###    endforeach()
###
###endif()
###
