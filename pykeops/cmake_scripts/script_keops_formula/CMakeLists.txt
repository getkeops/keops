########################################################################################################################
#                                                       HEADERS                                                        #
########################################################################################################################

cmake_minimum_required(VERSION 3.10)

project(PyKeOps LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

set(ignoreMe "${PYTORCH_ROOT_DIR}")

set(PYKEOPS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../)
set(KEOPS_SOURCE_DIR ${PYKEOPS_SOURCE_DIR}/keops/)
set(BIN_DIR ${PROJECT_BINARY_DIR}/../) # installation directory

## Set Path to sources
set(SOURCE_FILES
        ${PYKEOPS_SOURCE_DIR}
        ${PROJECT_BINARY_DIR}
		${KEOPS_SOURCE_DIR}
)

Include_Directories(${SOURCE_FILES})

include(${KEOPS_SOURCE_DIR}/cuda.cmake)

include(../PyKeOpsHeader.cmake)

if(${PYTHON_LANG} STREQUAL "torch")
    # fix for pytorch: https://discuss.pytorch.org/t/pytorch-0-4-1-undefined-symbol-at-import-of-a-cpp-extension/24420
    # and https://stackoverflow.com/questions/33394934/converting-std-cxx11string-to-stdstring
    if(NOT DEFINED _GLIBCXX_USE_CXX11_ABI)
        Set(_GLIBCXX_USE_CXX11_ABI 0)  # set default value to False...
    endif()
    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=${_GLIBCXX_USE_CXX11_ABI})
endif()


# - type for computation. The CACHE option enable to see it in ccmake.
if(NOT __TYPE__)
  Set(__TYPE__ float CACHE STRING "Precision type of the computations (__half, float or double)")
endif()
add_definitions(-D__TYPE__=${__TYPE__})

# - Choose if the multi-dimensional kernels are stored column or row wise
if(NOT C_CONTIGUOUS)
  Set(C_CONTIGUOUS O CACHE STRING "Multi-dimensional kernels are stored column wise.")
endif()
add_definitions(-DC_CONTIGUOUS=${C_CONTIGUOUS})

# this dummy flag is used in the bindings
if (${__TYPE__} STREQUAL "double")
    add_definitions(-DUSE_DOUBLE=1)
else ()
    add_definitions(-DUSE_DOUBLE=0)
endif ()

# this dummy flag is used in the bindings
if (${__TYPE__} STREQUAL "half2")
    add_definitions(-DUSE_HALF=1)
else ()
    add_definitions(-DUSE_HALF=0)
endif ()

# - some options for accuracy of summations
if(__TYPEACC__)
  add_definitions(-D__TYPEACC__=${__TYPEACC__})
endif()
if(SUM_SCHEME)
  add_definitions(-DSUM_SCHEME=${SUM_SCHEME})
endif()

# options for special computation schemes for large dimension
if(DEFINED ENABLECHUNK)
  add_definitions(-DENABLECHUNK=${ENABLECHUNK})
endif()
if(DEFINED DIM_TRESHOLD_CHUNK)
  add_definitions(-DDIM_TRESHOLD_CHUNK=${DIM_TRESHOLD_CHUNK})
endif()
if(DEFINED DIMCHUNK)
  add_definitions(-DDIMCHUNK=${DIMCHUNK})
endif()
if(DEFINED ENABLE_FINAL_CHUNKS)
  add_definitions(-DENABLE_FINAL_CHUNKS=${ENABLE_FINAL_CHUNKS})
endif()
if(DEFINED DIMFINALCHUNK)
  add_definitions(-DDIMFINALCHUNK=${DIMFINALCHUNK})
endif()
if(DEFINED MULT_VAR_HIGHDIM)
  add_definitions(-DMULT_VAR_HIGHDIM=${MULT_VAR_HIGHDIM})
endif()

########################################################################################################################
#                                                        KEOPS                                                         #
########################################################################################################################

if (USE_CUDA) # ----------------- create shared lib (cuda)

    add_library(
            copy_${shared_obj_name} OBJECT
            ${KEOPS_SOURCE_DIR}/core/link_autodiff.cu
    )

    # tell Cmake to explicitly add the dependency: keops is recompiled as soon as formula.h changes.
    set_source_files_properties(
        ${KEOPS_SOURCE_DIR}/core/link_autodiff.cu PROPERTIES
        OBJECT_DEPENDS ${shared_obj_name}.h
    )

else () # ----------------- create shared lib (cpp)

    add_library(
            copy_${shared_obj_name} OBJECT
            ${KEOPS_SOURCE_DIR}/core/link_autodiff.cpp
    )

    # tell Cmake to explicitly add the dependency: keops is recompiled as soon as formula.h changes.
    set_source_files_properties(
        ${KEOPS_SOURCE_DIR}/core/link_autodiff.cpp PROPERTIES
        OBJECT_DEPENDS ${shared_obj_name}.h
    )

endif ()

target_compile_options(
        copy_${shared_obj_name} BEFORE
        PRIVATE -include ${shared_obj_name}.h
)

add_custom_target(
        ${shared_obj_name}
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_OBJECTS:copy_${shared_obj_name}> ${BIN_DIR}/${shared_obj_name}.o
)

add_dependencies(${shared_obj_name} copy_${shared_obj_name})

# Write a log file to decypher keops dllname
include(../PyKeOpsLog.cmake)


