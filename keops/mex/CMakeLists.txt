set(MATLAB_ADDITIONAL_VERSIONS
    "R2007b=7.5"
    "R2007a=7.4"
)
find_package(Matlab)

### - mexfiles : generate a working mexfile is painful with cmake. 
##   The problem is twofold : the cuda module do not define a 
##   CUDA_add_object. We then need to use Add_library instead.

if(Matlab_FOUND)

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


    Add_library( # generate the string "g++ -c ..." but do not execute it
        mex_file_cpp OBJECT
        ${CMAKE_CURRENT_SOURCE_DIR}/cudaconv.cpp
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
            COMMAND ${Matlab_ROOT_DIR}/bin/mex ${rpat} $<TARGET_FILE:keops> ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/mex_file_cpp.dir/cudaconv.cpp.o -output ${mex_name} 
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



    #matlab_add_mex(
        #NAME mex_cpp2
        #SRC ${CMAKE_CURRENT_SOURCE_DIR}/cudaconv.cpp
        #OUTPUT_NAME ${mex_name}_cpp
        #)

    #target_compile_options(
        #mex_cpp2 BEFORE
        #PRIVATE -include formula.h 
        #)

    #set_target_properties( # pass needed options to add_custom_target()
        #mex_cpp2 PROPERTIES 
        #PREFIX "" 
        #LINKER_LANGUAGE CXX
        #EXCLUDE_FROM_ALL TRUE
        #)
else()
    message(STATUS "   ==> No mex file can be built.")
endif()

