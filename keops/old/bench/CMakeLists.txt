find_package(Threads)

#------------------------------------------------------------------------------------#
#-----------------------------FIND GOOGLE BENCHMARK----------------------------------#
#------------------------------------------------------------------------------------#

if(Threads_FOUND) # skip bench if pthread is not found

    #------------------------------------------------------------------------------------#
    #----------------------------------COMPILATIONS--------------------------------------#
    #------------------------------------------------------------------------------------#
    
    if (USE_CUDA)
    
        add_executable(
            bench_grad1conv
            ${CMAKE_CURRENT_SOURCE_DIR}/bench_grad1conv.cu
        )
    
        target_link_libraries(
            bench_grad1conv
            radial_kernels_grad1conv
            benchmark 
        )

    endif()
    
endif() # Threads_FOUND
