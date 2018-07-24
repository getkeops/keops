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
)

if(PYTORCH_INCLUDE_DIR)
    list(APPEND SOURCE_FILES ${PYTORCH_INCLUDE_DIR})
endif()

Include_Directories(${SOURCE_FILES})

include(${PROJECT_SRC}/headers.cmake)


########################################################################################################
#                                             PYBIND11                                                 #
########################################################################################################

add_subdirectory(pybind11)
add_definitions(-DMODULE_NAME=${shared_obj_name})
########################################################################################################
#                                        Generic                                                       #
########################################################################################################

# this dummy flag is used in the bindings
if (${__TYPE__} STREQUAL "double")
    add_definitions(-DUSE_DOUBLE=1)
else()
    add_definitions(-DUSE_DOUBLE=0)
endif()


# ----------------- create shared lib (cuda)
if(USE_CUDA)
  
    CUDA_add_library(
        keops${shared_obj_name} SHARED
        ${PROJECT_SRC}/core/link_autodiff.cu
        OPTIONS --pre-include=${shared_obj_name}.h
    )

else()
# ----------------- create shared lib (cpp)

    add_library(
        keops${shared_obj_name} SHARED
        ${PROJECT_SRC}/core/link_autodiff.cpp
    )

    target_compile_options(
        keops${shared_obj_name} BEFORE
        PRIVATE -include ${shared_obj_name}.h
    )

    # tell Cmake to explicitly add the dependency: keops is recompiled as soon as formula.h changes.
    set_source_files_properties(
        ${PROJECT_SRC}/core/link_autodiff.cpp PROPERTIES
        OBJECT_DEPENDS ${shared_obj_name}.h
    )

endif()

# set name
set_target_properties(keops${shared_obj_name} PROPERTIES
    LIBRARY_OUTPUT_NAME  keops${shared_obj_name}
    PREFIX ""
)

pybind11_add_module(${shared_obj_name}
    ${CMAKE_CURRENT_SOURCE_DIR}/common/keops_io.cpp
    )

target_compile_options(
        ${shared_obj_name} BEFORE
        PRIVATE  -include ${shared_obj_name}.h
    )

target_link_libraries(
    ${shared_obj_name} PUBLIC
    keops${shared_obj_name}
)


########################################################################################################
#                                        Specific                                                      #
########################################################################################################

if (USE_CUDA)


    #---------------------------------------------------------------------------------------------------
    CUDA_add_library(
        radial_kernel_conv_cuda SHARED
        ${PROJECT_SRC}/specific/radial_kernels/cuda_conv.cu
    )

    pybind11_add_module(radial_kernel_conv
        ${CMAKE_CURRENT_SOURCE_DIR}/numpy/convolutions/radial_kernel_conv.cpp
    )

    target_compile_options(
        radial_kernel_conv BEFORE
        PRIVATE  -include ${shared_obj_name}.h
    )

    target_link_libraries(
        radial_kernel_conv PUBLIC
        radial_kernel_conv_cuda
    )

    #---------------------------------------------------------------------------------------------------
    CUDA_add_library(
        radial_kernel_grad1conv_cuda SHARED
        ${PROJECT_SRC}/specific/radial_kernels/cuda_grad1conv.cu
    )

    pybind11_add_module(radial_kernel_grad1conv
        ${CMAKE_CURRENT_SOURCE_DIR}/numpy/convolutions/radial_kernel_grad1conv.cpp
    )

    target_compile_options(
        radial_kernel_grad1conv BEFORE
        PRIVATE  -include ${shared_obj_name}.h
    )

    target_link_libraries(
        radial_kernel_grad1conv PUBLIC
        radial_kernel_grad1conv_cuda
    )

    #---------------------------------------------------------------------------------------------------
    if(NOT KERNEL_GEOM OR (KERNEL_GEOM STREQUAL "gaussian"))
        SET(KERNEL_GEOM "gaussian")
        SET(KERNEL_GEOM_TYPE 0)
    elseif(KERNEL_GEOM STREQUAL "cauchy")
            SET(KERNEL_GEOM_TYPE 1)
    else()
        message(FATAL_ERROR "Set KERNEL_GEOM type to gaussian or cauchy.")
    endif()
    add_definitions(-DKERNEL_GEOM_TYPE=${KERNEL_GEOM_TYPE})

    if(NOT KERNEL_SIG OR (KERNEL_SIG STREQUAL gaussian))
        SET(KERNEL_SIG "gaussian")
        SET(KERNEL_SIG_TYPE 0)
    elseif(KERNEL_SIG STREQUAL cauchy)
        SET(KERNEL_SIG_TYPE 1)
    else()
        message(FATAL_ERROR "Set KERNEL_SIG type to gaussian or cauchy.")
    endif()
    add_definitions(-DKERNEL_SIG_TYPE=${KERNEL_SIG_TYPE})

    if(NOT KERNEL_SPHERE OR (KERNEL_SPHERE STREQUAL gaussian_unoriented))
        SET(KERNEL_SPHERE "gaussian_unoriented")
        SET(KERNEL_SPHERE_TYPE 0)
    elseif(KERNEL_SPHERE STREQUAL binet)
        SET(KERNEL_SPHERE_TYPE 1)
    elseif(KERNEL_SPHERE STREQUAL gaussian_oriented)
        SET(KERNEL_SPHERE_TYPE 2)
    elseif(KERNEL_SPHERE STREQUAL linear)
        SET(KERNEL_SPHERE_TYPE 3)
    else()
        message(FATAL_ERROR "Set KERNEL_SPHERE type to gaussian_unoriented, binet, gaussian_oriented or linear.")
    endif()
    add_definitions(-DKERNEL_SPHERE_TYPE=${KERNEL_SPHERE_TYPE})

    #foreach(ext_name "" "_dx" "_df" "_dxi")
    foreach(ext_name "")

        SET(fshape_scp_name fshape_scp${ext_name}_${KERNEL_GEOM}${KERNEL_SIG}${KERNEL_SPHERE}_${__TYPE__})

        SET(name1 fshape_gpu${ext_name})
        CUDA_add_library(
            ${name1} SHARED
            ${PROJECT_SRC}/specific/shape_distance/${name1}.cu
        )
        set_target_properties(${name1} PROPERTIES
            LIBRARY_OUTPUT_NAME ${fshape_scp_name}
            PREFIX ""
        )

        SET(name2 fshape_scp${ext_name})
        add_definitions(-DMODULE_NAME_FSHAPE_SCP=${fshape_scp_name})

        pybind11_add_module(${fshape_scp_name}
            ${CMAKE_CURRENT_SOURCE_DIR}/numpy/shape_distance/${name2}.cpp
        )

        target_compile_options(
            ${fshape_scp_name} BEFORE
            PRIVATE  -include ${shared_obj_name}.h
        )

        target_link_libraries(
            ${fshape_scp_name} PUBLIC
            ${name1}
        )

    endforeach()

endif()