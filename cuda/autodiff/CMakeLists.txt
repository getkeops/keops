#------------------------------------------------------------------------------------#
#------------------------------------HEADERS-----------------------------------------#
#------------------------------------------------------------------------------------#

cmake_minimum_required(VERSION 3.9)

project(libkp LANGUAGES CXX)

## Set Path to sources
set(SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}

    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/core/formulas
    ${CMAKE_CURRENT_SOURCE_DIR}/core/reductions
    ${CMAKE_CURRENT_SOURCE_DIR}/test
    ${CMAKE_CURRENT_SOURCE_DIR}/sandbox
    ${PROJECT_BINARY_DIR}
)

Include_Directories(${SOURCE_FILES})

set (CMAKE_CXX_FLAGS "--std=c++11 -O3")

set (CUDA_PROPAGATE_HOST_FLAGS ON)




#------------------------------------------------------------------------------------#
#------------------------------FIND CUDA AND GPUs------------------------------------#
#------------------------------------------------------------------------------------#

find_package(CUDA)

# A function for automatic detection of GPUs installed (source: caffe git repo).
function(caffe_detect_installed_gpus out_variable)
    if(NOT CUDA_gpu_detect_output)
        set(__cufile ${PROJECT_BINARY_DIR}/detect_cuda_archs.cu)

        file(WRITE ${__cufile} ""
            "#include <cstdio>\n"
            "int main()\n"
            "{\n"
            "  int count = 0;\n"
            "  if (cudaSuccess != cudaGetDeviceCount(&count)) return -1;\n"
            "  if (count == 0) return -1;\n"
            "  for (int device = 0; device < count; ++device)\n"
            "  {\n"
            "    cudaDeviceProp prop;\n"
            "    if (cudaSuccess == cudaGetDeviceProperties(&prop, device))\n"
            "      std::printf(\"%d.%d \", prop.major, prop.minor);\n"
            "  }\n"
            "  return 0;\n"
            "}\n")

        execute_process(COMMAND "${CUDA_NVCC_EXECUTABLE}" "--run" "${__cufile}"
            WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/CMakeFiles/"
            RESULT_VARIABLE __nvcc_res OUTPUT_VARIABLE __nvcc_out
            ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

        if(__nvcc_res EQUAL 0)
            string(REPLACE "2.1" "2.1(2.0)" __nvcc_out "${__nvcc_out}")
            set(CUDA_gpu_detect_output ${__nvcc_out} CACHE INTERNAL "Returned GPU architetures from caffe_detect_gpus tool" FORCE)
        endif()
    endif()

    if(NOT CUDA_gpu_detect_output)
        message(STATUS "Automatic GPU detection failed. Building for all known architectures.")
        set(${out_variable}  "20 21(20) 30 35 50 60 61" PARENT_SCOPE)
    else()
        set(${out_variable} ${CUDA_gpu_detect_output} PARENT_SCOPE)
    endif()
endfunction()


# run the detection 
if(NOT gpu_compute_capability)
    caffe_detect_installed_gpus(gpu_compute_capability)

    # remove dots and convert to lists
    string(REGEX REPLACE "\\." "" gpu_compute_capability "${gpu_compute_capability}")
    string(REGEX MATCHALL "[0-9()]+" gpu_compute_capability "${gpu_compute_capability}")
    List(REMOVE_DUPLICATES gpu_compute_capability)

    message(STATUS "Compute capability automatically set to: ${gpu_compute_capability}")
else()

    message(STATUS "Compute capability manually set to ${gpu_compute_capability}")
endif()

# Options for nvcc
List(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_${gpu_compute_capability},code=sm_${gpu_compute_capability}")
List(APPEND CUDA_NVCC_FLAGS "--use_fast_math")
List(APPEND CUDA_NVCC_FLAGS "--compiler-options=-fPIC")





#------------------------------------------------------------------------------------#
#--------------------------------Programm Options------------------------------------#
#------------------------------------------------------------------------------------#

# Template macros.
add_definitions(-D_FORCE_INLINES)

# - type for computation. The CACHE option enable to see it in ccmake.
if(NOT __TYPE__)
    Set(__TYPE__ float CACHE STRING "Precision type of the computations (float or double)")
endif()


# - Declare the templates formula if not provided by the user
if(NOT USENEWSYNTAX)

    if(NOT FORMULA)
        Set(FORMULA "GaussKernel<_P<0>,_X<0,3>,_Y<0,3>,_Y<1,3>>" CACHE STRING "Template formula to be instantiate")
    endif()
    unset(FORMULA_OBJ CACHE)

else()
    
    if(NOT FORMULA_OBJ)
        Set(VAR_ALIASES "auto x=Vx(0,3); auto y=Vy(1,3); auto u=Vx(2,4); auto v=Vy(3,4); auto b=Vy(4,3); auto p=Pm(0);")
        Set(FORMULA_OBJ "Square((u,v))*Exp(-Cst(p)*SqNorm2(x-y))*b")
    endif()
    unset(FORMULA CACHE)

endif()

# We should generate a file to avoid parsing problem with shell: write the macros  in a file which will be included
configure_file(template.h.in template.h @ONLY)

# unset all options to prevent cache problem: the configure_file will be rebuild every time
unset(USENEWSYNTAX CACHE)
unset(FORMULA CACHE)
unset(FORMULA_OBJ CACHE)
unset(__TYPE__ CACHE)
unset(VAR_ALIASES CACHE)




#------------------------------------------------------------------------------------#
#----------------------------------COMPILATIONS--------------------------------------#
#------------------------------------------------------------------------------------#

    
# ----------------- create shared lib (cuda)

if(CUDA_FOUND)
  
CUDA_add_library(
        shared_obj SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/core/link_autodiff.cu
        OPTIONS --pre-include=template.h
    )


    # shared_lib can be rename if the variable shared_lib_name is provided.
    if(shared_obj_name)
        add_custom_command(
            TARGET shared_obj POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E rename
            $<TARGET_FILE:shared_obj>
            ${shared_obj_name}    
        )
    endif()

endif()



# ----------------- create shared lib (cpp)

add_library(
    shared_obj_cpp SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/core/link_autodiff.cpp
)

target_compile_options(
    shared_obj_cpp BEFORE
    PRIVATE -include template.h
)





# ----------------- test link
if (CUDA_FOUND)

    CUDA_add_executable(test_link_conv #${test_link})
        ${CMAKE_CURRENT_SOURCE_DIR}/sandbox/test_link.cu
        OPTIONS -D_MWAITXINTRIN_H_INCLUDED
        OPTIONS --pre-include=template.h
    )

    target_link_libraries(
        test_link_conv
        shared_obj
    )

endif()

# ----------------- various targets

add_subdirectory(bench)
add_subdirectory(test)
add_subdirectory(matlab_bindings)
add_subdirectory(sandbox)


