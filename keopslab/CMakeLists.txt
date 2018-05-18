#------------------------------------------------------------------------------------#
#------------------------------------HEADERS-----------------------------------------#
#------------------------------------------------------------------------------------#

cmake_minimum_required(VERSION 2.8)

project(KeOpsLab LANGUAGES CXX)

## Set Path to sources
set(PROJECT_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../keops)

set(SOURCE_FILES
    ${PROJECT_SRC}
    ${PROJECT_BINARY_DIR}
)

Include_Directories(${SOURCE_FILES})

include(${PROJECT_SRC}/headers.cmake)

#------------------------------------------------------------------------------------#
#----------------------------------COMPILATIONS--------------------------------------#
#------------------------------------------------------------------------------------#




set(MATLAB_ADDITIONAL_VERSIONS
    "R2007b=7.5"
    "R2007a=7.4"
)
find_package(Matlab)

### - mexfiles : generate a working mexfile is painful with cmake. 
##   The problem is twofold : the cuda module do not define a 
##   CUDA_add_object. We then need to use Add_library instead.

if(Matlab_FOUND)

    # ----------------- get some env variables

    # this dummy flag is used to cast the input array of Matlab
    if (${__TYPE__} STREQUAL "double")
        add_definitions(-DUSE_DOUBLE)
    endif()

    if(NOT DEFINED mex_name)
        Set(mex_name mex_binding)
    endif()

    matlab_get_mex_suffix( # simply get the extension : almost surely mexa64
        ${Matlab_ROOT_DIR} 
        mex_suffix
    )

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

    # shared_lib can be rename if the variable shared_lib_name is provided.
    if(shared_obj_name)

        set_target_properties(keops PROPERTIES
            LIBRARY_OUTPUT_NAME  ${shared_obj_name}
            PREFIX ""
        )

    endif()



    # ----------------- create mex files generic

    Add_library( # generate the string "g++ -c ..." but do not execute it
        mex_file_cpp OBJECT
        ${CMAKE_CURRENT_SOURCE_DIR}/generic/cudaconv.cpp
    )

    target_compile_options(
        mex_file_cpp BEFORE
        PRIVATE -I${Matlab_INCLUDE_DIRS} -fPIC
    )

    set(rpat LDFLAGS=\"$$LDFLAGS -Wl,-rpath,\\\\\\$$ORIGIN\")

    if(NOT(${CMAKE_VERSION} VERSION_LESS 3.8.0))
        add_custom_target(
            mex_cpp
            DEPENDS keops mex_file_cpp # ensure obj file is created before using mex for linking
            COMMAND ${Matlab_ROOT_DIR}/bin/mex ${rpat} $<TARGET_FILE:keops> $<TARGET_OBJECTS:mex_file_cpp> -output ${mex_name} # only since cmake 3.9 
        )
    else()
        add_custom_target(
            mex_cpp
            DEPENDS keops mex_file_cpp # ensure obj file is created before using mex for linking
            COMMAND ${Matlab_ROOT_DIR}/bin/mex ${rpat} $<TARGET_FILE:keops> ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/mex_file_cpp.dir/generic/cudaconv.cpp.o -output ${mex_name} 
        )
    endif()

    add_dependencies(
        mex_cpp
        keops
    )

    set_target_properties( # pass needed options to add_custom_target()
        mex_cpp PROPERTIES 
        PREFIX "" 
        LINKER_LANGUAGE CXX
        EXCLUDE_FROM_ALL FALSE
    )


    # ----------------- create mex files specific
    if(USE_CUDA)
      
        if(NOT KERNEL_GEOM OR (KERNEL_GEOM STREQUAL "gaussian"))
            SET(KERNEL_GEOM_TYPE 0)
        elseif(KERNEL_GEOM STREQUAL "cauchy")
            SET(KERNEL_GEOM_TYPE 1)
        else()
            message(FATAL_ERROR "Set KERNEL_GEOM type to gaussian or cauchy.")
        endif()
        add_definitions(-DKERNEL_GEOM_TYPE=${KERNEL_GEOM_TYPE})

				# enlevé la ligne suivante car ça crée une erreur sur une de mes machines ("message called with incorrect number of arguments")
        #message(${KERNEL_SIG})

        if(NOT KERNEL_SIG OR (KERNEL_SIG STREQUAL gaussian))
            SET(KERNEL_SIG_TYPE 0)
        elseif(KERNEL_SIG STREQUAL cauchy)
            SET(KERNEL_SIG_TYPE 1)
        else()
            message(FATAL_ERROR "Set KERNEL_SIG type to gaussian or cauchy.")
        endif()
        add_definitions(-DKERNEL_SIG_TYPE=${KERNEL_SIG_TYPE})

        if(NOT KERNEL_SPHERE OR (KERNEL_SPHERE STREQUAL gaussian_unoriented))
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


        # ----------------- fshape_scp_dx
        foreach(ext_name "" "_dx" "_df" "_dxi")

            SET(mex_fshape_scp_name cuda_fshape_scp${ext_name}_${KERNEL_GEOM}${KERNEL_SIG}${KERNEL_SPHERE})

            SET(name1 fshape_gpu${ext_name})
            CUDA_add_library(
                ${name1} SHARED
                ${PROJECT_SRC}/specific/shape_distance/${name1}.cu
            )
            set_target_properties(${name1} PROPERTIES
                LIBRARY_OUTPUT_NAME ${mex_fshape_scp_name}
                PREFIX ""
            )

            SET(name2 cudafshape${ext_name})
            Add_library( # generate the string "g++ -c ..." but do not execute it
                ${name2} OBJECT
                ${CMAKE_CURRENT_SOURCE_DIR}/specific/shape_distance/${name2}.cpp
            )
            target_compile_options(
                ${name2} BEFORE
                PRIVATE -I${Matlab_INCLUDE_DIRS} -fPIC
            )

            SET(name3 mex_fshape_scp${ext_name})
            if(NOT(${CMAKE_VERSION} VERSION_LESS 3.8.0))
                add_custom_target(
                    ${name3}
                    DEPENDS ${name1} ${name2}# ensure obj file is created before using mex for linking
                    COMMAND ${Matlab_ROOT_DIR}/bin/mex ${rpat} $<TARGET_FILE:${name1}> $<TARGET_OBJECTS:${name2}> -output ${mex_fshape_scp_name} # only since cmake 3.9 
                )
            else()
                add_custom_target(
                    ${name3}
                    DEPENDS ${name1} ${name2}# ensure obj file is created before using mex for linking
                    COMMAND ${Matlab_ROOT_DIR}/bin/mex ${rpat} $<TARGET_FILE:${name1}> ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${name2}.dir/specific/shape_distance/cudafshape${ext_name}.cpp.o -output ${mex_fshape_scp_name}
                )
            endif()

            set_target_properties( # pass needed options to add_custom_target()
                ${name3} PROPERTIES 
                PREFIX "" 
                LINKER_LANGUAGE CXX
                EXCLUDE_FROM_ALL TRUE
                )
    endforeach()

    endif()
else()
    message(STATUS "   ==> No mex file can be built.")
endif()

