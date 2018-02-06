find_package(Matlab)

### - mexfiles : generate a working mexfile is painful with cmake. 
    ##   The problem is twofold : the cuda module do not define a 
    ##   CUDA_add_object. We then need to use Add_library instead.

if(Matlab_FOUND)

    if(NOT mex_name)
        Set(mex_name essai_mex)
    endif()
    #matlab_get_mex_suffix( # simply get the extension : almost surely mexa64
        #${Matlab_ROOT_DIR} 
        #mex_suffix
    #)
    
    if(CUDA_FOUND)

        CUDA_compile( # generate the string "nvcc -c ..." but do not execute it
            mex_file_cuda SHARED
            ${CMAKE_CURRENT_SOURCE_DIR}/cudaconv.cpp
            OPTIONS --pre-include=template.h -I${Matlab_INCLUDE_DIRS} -x cu
        )

        set_target_properties( # pass needed options to add_custom_target()
            mex_file_cuda PROPERTIES 
            PREFIX "" 
            LINKER_LANGUAGE CXX
            EXCLUDE_FROM_ALL TRUE
        )

        matlab_add_mex(
            NAME mex_cuda SHARED
            SRC $<TARGET_FILE:mex_file_cuda>
            OUTPUT_NAME ${mex_name}
            LINK_TO cudart
            )

        #add_custom_target(
            #mex_cuda
            #DEPENDS ${mex_file} # generate the .o as it execute the command made by CUDA_compile
        #)

        #add_custom_command( # link the .o by running mex compiler
            #TARGET mex_cuda POST_BUILD
            #COMMAND ${Matlab_ROOT_DIR}/bin/mex  -L${CUDA_TOOLKIT_ROOT_DIR}/lib64/ -lcudart -L${Matlab_LIBRARIES} -lmex ${mex_file} -output tmp
            #COMMAND ${CMAKE_COMMAND} -E rename tmp.${mex_suffix} \"${FORMULA}\"_${__TYPE__}.${mex_suffix}
        #)

        #set_target_properties( # pass needed options to add_custom_target()
            #mex_cuda PROPERTIES 
            #PREFIX "" 
            #LINKER_LANGUAGE CXX
            #EXCLUDE_FROM_ALL TRUE
        #)

    else()

        Add_library( # generate the string "g++ -c ..." but do not execute it
            mex_file_cpp OBJECT
            ${CMAKE_CURRENT_SOURCE_DIR}/cudaconv.cpp
        )

        set_target_properties( # pass needed options to add_custom_target()
            mex_file_cpp PROPERTIES 
            PREFIX "" 
            LINKER_LANGUAGE CXX
            EXCLUDE_FROM_ALL TRUE
        )

        target_compile_options(
            mex_file_cpp BEFORE
            PRIVATE -include template.h -I${Matlab_INCLUDE_DIRS} -fPIC
        )

        matlab_add_mex(
            NAME mex_cpp SHARED
            SRC $<TARGET_OBJECTS:mex_file_cpp>
            OUTPUT_NAME ${mex_name}
            )

        #add_custom_target(
            #mex_cpp
            #DEPENDS mex_file_cpp # generate the .o as it execute the command made by CUDA_compile
        #)
    
         #add_custom_command( # link the .o by running mex compiler
            #TARGET mex_cpp POST_BUILD
            #COMMAND ${Matlab_ROOT_DIR}/bin/mex $<TARGET_OBJECTS:mex_file_cpp> -output tmp
        #)
        
        #if(mex_name)
            #add_custom_command(
                #TARGET mex_cpp POST_BUILD
                #COMMAND ${CMAKE_COMMAND} -E rename ${CMAKE_CURRENT_BINARY_DIR}/tmp.${mex_suffix} ${CMAKE_CURRENT_BINARY_DIR}/${mex_name}.${mex_suffix}
            #)
        #endif()

    endif()

else()
    message(STATUS "   ==> No mex file can be build.")
endif()

