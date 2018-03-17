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

    set(rpat LDFLAGS=\"-Wl,-rpath,\\\\\\$$ORIGIN\")

    if(NOT(${CMAKE_VERSION} VERSION_LESS 3.8.0))
        add_custom_target(
            mex_cpp
            DEPENDS keops mex_file_cpp # ensure obj file is created before using mex for linking
            COMMAND ${Matlab_ROOT_DIR}/bin/mex ${rpat} $<TARGET_FILE:keops> $<TARGET_OBJECTS:mex_file_cpp>  -output ${mex_name} # only since cmake 3.9 
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


    # ----------------- create mex files generic

    matlab_add_mex(
        NAME mex_specific
        SRC ${CMAKE_CURRENT_SOURCE_DIR}/specific/shape_distance/cudafshape.cpp
        OUTPUT_NAME ${mex_name}
        )

    #target_compile_options(
        #mex_cpp2 BEFORE
        #PRIVATE -include formula.h 
        #)

    set_target_properties( # pass needed options to add_custom_target()
        mex_specific PROPERTIES 
        PREFIX "" 
        LINKER_LANGUAGE CXX
        EXCLUDE_FROM_ALL TRUE
        )
else()
    message(STATUS "   ==> No mex file can be built.")
endif()

