find_package(Matlab)

### - mexfiles : generate a working mexfile is painful with cmake. 
##   The problem is twofold : the cuda module do not define a 
##   CUDA_add_object. We then need to use Add_library instead.

if(Matlab_FOUND AND NOT(${CMAKE_VERSION} VERSION_LESS 3.8.0))

    # this dummy flag is used to cast the input array of Matlab
    if (${__TYPE__} STREQUAL "double")
        add_definitions(-DUSE_DOUBLE)
    endif()

    if (CUDA_FOUND)
        add_definitions(-DUSE_CUDA)
    endif()

    if(NOT mex_name)
        Set(mex_name mex_binding)
    endif()

    if(shared_obj_name)

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
        PRIVATE -include formula.h -I${Matlab_INCLUDE_DIRS} -fPIC
    )

    if(NOT mex_name)
        Set(mex_name last_mex) 
    endif()


    set(rpat LDFLAGS=\"-Wl,-rpath=\\\\\\$$ORIGIN\")
    add_custom_target(
        mex_cpp
        DEPENDS shared_obj mex_file_cpp # ensure obj file is created before using mex for linking
        COMMAND ${Matlab_ROOT_DIR}/bin/mex ${rpat} $<TARGET_FILE:shared_obj> $<TARGET_OBJECTS:mex_file_cpp>  -output ${mex_name} # only since cmake 3.9 
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

