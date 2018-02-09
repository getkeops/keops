find_package(Matlab)

### - mexfiles : generate a working mexfile is painful with cmake. 
##   The problem is twofold : the cuda module do not define a 
##   CUDA_add_object. We then need to use Add_library instead.

if(${__TYPE__} EQUAL "double" AND Matlab_FOUND AND NOT(${CMAKE_VERSION} VERSION_LESS 3.8.0))

    if(NOT mex_name)
        Set(mex_name mex_binding)
    endif()

    matlab_get_mex_suffix( # simply get the extension : almost surely mexa64
        ${Matlab_ROOT_DIR} 
        mex_suffix
    )

    if(CUDA_FOUND)

        CUDA_compile( # generate the string "nvcc -c ..." but do not execute it
            mex_file_cuda SHARED
            ${CMAKE_CURRENT_SOURCE_DIR}/cudaconv.cu
            OPTIONS --pre-include=formula.h -I${Matlab_INCLUDE_DIRS}
        )

        add_custom_target(
            mex_cuda ALL
            DEPENDS ${mex_file_cuda} # generate the .o as it execute the command made by CUDA_compile
        )

    add_custom_command( # link the .o by running mex compiler. Note : -L${CUDA_TOOLKIT_ROOT_DIR}/lib64/ should be pass explicitly...
            TARGET mex_cuda POST_BUILD
            COMMAND ${Matlab_ROOT_DIR}/bin/mex  -L${CUDA_TOOLKIT_ROOT_DIR}/lib64/ -lcudart -L${Matlab_LIBRARIES} -lmex ${mex_file_cuda} -output tmp
        )
        
        if(mex_name)
        add_custom_command(
            TARGET mex_cuda POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E rename ${CMAKE_CURRENT_BINARY_DIR}/tmp.${mex_suffix} ${CMAKE_CURRENT_BINARY_DIR}/${mex_name}_cuda.${mex_suffix}
        )
        endif()

        set_target_properties( # pass needed options to add_custom_target()
             mex_cuda PROPERTIES 
            PREFIX "" 
            LINKER_LANGUAGE CXX
            EXCLUDE_FROM_ALL FALSE
        )

        # solution using FindMatlab seems to be broken...
        #matlab_add_mex(
            #NAME mex_cuda
            #SRC ${mex_file_cuda}
            #OUTPUT_NAME ${mex_name}
        #)

    endif()

    Add_library( # generate the string "g++ -c ..." but do not execute it
        mex_file_cpp OBJECT
        ${CMAKE_CURRENT_SOURCE_DIR}/cudaconv.cpp
    )

    target_compile_options(
        mex_file_cpp BEFORE
        PRIVATE -include formula.h -I${Matlab_INCLUDE_DIRS} -fPIC
    )

    add_custom_target(
        mex_cpp
        DEPENDS mex_file_cpp # ensure obj file is created before using mex for linking
        COMMAND ${Matlab_ROOT_DIR}/bin/mex $<TARGET_OBJECTS:mex_file_cpp> -output tmp # only since cmake 3.9
    )

    if(mex_name)
        add_custom_command(
            TARGET mex_cpp POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E rename ${CMAKE_CURRENT_BINARY_DIR}/tmp.${mex_suffix} ${CMAKE_CURRENT_BINARY_DIR}/${mex_name}_cpp.${mex_suffix}
        )
    endif()

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

