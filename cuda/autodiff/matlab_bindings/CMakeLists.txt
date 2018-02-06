find_package(Matlab)

### - mexfiles : generate a working mexfile is painful with cmake. 
    ##   The problem is twofold : the cuda module do not define a 
    ##   CUDA_add_object. We then need to use Add_library instead.

if(Matlab_FOUND)

    CUDA_compile( # generate the string "nvcc -c ..." but do not execute it
        mex_file SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/cudaconv.cu
        OPTIONS --pre-include=template.h -I${Matlab_INCLUDE_DIRS} -x cu
    )

    add_custom_target(
        mex_cuda ALL
        DEPENDS ${mex_file} # generate the .o as it execute the command made by CUDA_compile
    )

    matlab_get_mex_suffix( # simply get the extension : almost surely mexa64
        ${Matlab_ROOT_DIR} 
        mex_suffix
    )

    add_custom_command( # link the .o by running mex compiler
        TARGET mex_cuda POST_BUILD
        COMMAND ${Matlab_ROOT_DIR}/bin/mex  -L${CUDA_TOOLKIT_ROOT_DIR}/lib64/ -lcudart -L${Matlab_LIBRARIES} -lmex ${mex_file} -output tmp
        COMMAND ${CMAKE_COMMAND} -E rename tmp.${mex_suffix} \"${FORMULA}\"_${__TYPE__}.${mex_suffix}
    )

    set_target_properties( # pass needed options to add_custom_target()
        mex_cuda PROPERTIES 
        PREFIX "" 
        LINKER_LANGUAGE CXX
        EXCLUDE_FROM_ALL TRUE
    )
else()
    message(STATUS "   ==> No mex file can be build.")
endif()


