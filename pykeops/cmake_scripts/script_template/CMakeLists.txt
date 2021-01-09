########################################################################################################################
#                                                       HEADERS                                                        #
########################################################################################################################

cmake_minimum_required(VERSION 3.10)

project(PyKeOps LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

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

# - type for computation. The CACHE option enable to see it in ccmake.
if(NOT __TYPE__)
  Set(__TYPE__ float CACHE STRING "Precision type of the computations (__half, float or double)")
endif()
add_definitions(-D__TYPE__=${__TYPE__})

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

# - Choose if the multi-dimensional kernels are stored column or row wise
if(NOT C_CONTIGUOUS)
  Set(C_CONTIGUOUS O CACHE STRING "Multi-dimensional kernels are stored column wise.")
endif()
add_definitions(-DC_CONTIGUOUS=${C_CONTIGUOUS})

########################################################################################################################
#                                                       PYTORCH                                                        #
########################################################################################################################

if (NOT DEFINED PYTHON_LANG)
    Set(PYTHON_LANG numpy)
endif ()

if (${PYTHON_LANG} STREQUAL "torch")

    # The  following lines could replace the included files below. Unfortunately, the TorchConfig.cmake import too many
    # library (caffe2 ...) which is not relevent in our case.
    #set(CMAKE_PREFIX_PATH ${PYTORCH_ROOT_DIR})
    #find_package(Torch REQUIRED)
    #add_definitions(${TORCH_CXX_FLAGS})

    Include_Directories(
            ${PYTORCH_ROOT_DIR}/include/
            ${PYTORCH_ROOT_DIR}/include/torch/csrc/api/include/
    )

    # fix for pytorch: https://discuss.pytorch.org/t/pytorch-0-4-1-undefined-symbol-at-import-of-a-cpp-extension/24420
    # and https://stackoverflow.com/questions/33394934/converting-std-cxx11string-to-stdstring
    if(NOT DEFINED _GLIBCXX_USE_CXX11_ABI)
        Set(_GLIBCXX_USE_CXX11_ABI 0)  # set default value to False...
    endif()
    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=${_GLIBCXX_USE_CXX11_ABI})

    # We should generate a file to avoid parsing problem with shell: write the macros in a file which will be included
    configure_file(${PYKEOPS_SOURCE_DIR}/torch_headers.h.in ${CMAKE_CURRENT_BINARY_DIR}/torch_headers.h @ONLY)
endif()


########################################################################################################################
#                                                     PYBIND11                                                         #
########################################################################################################################

add_subdirectory(${PYKEOPS_SOURCE_DIR}/pybind11 ${CMAKE_CURRENT_BINARY_DIR}/pybind11)  #find_package(pybind11  REQUIRED)
add_definitions(-DMODULE_NAME=${template_name})

pybind11_add_module(${template_name}
        ${PYKEOPS_SOURCE_DIR}/${PYTHON_LANG}/generic/generic_red.cpp
		${keops_formula_name}.o
)

if (USE_CUDA)
	set_target_properties(${template_name} PROPERTIES LINKER_LANGUAGE CUDA) 
	target_link_libraries(${template_name} PUBLIC ${CUDA_LIBRARIES})
endif()

SET_SOURCE_FILES_PROPERTIES(
  ${BIN_DIR}/${keops_formula_name}.o
  PROPERTIES
  EXTERNAL_OBJECT true
  GENERATED true
)


if(${PYTHON_LANG} STREQUAL "torch")

    set_source_files_properties(
            ${PYKEOPS_SOURCE_DIR}/${PYTHON_LANG}/generic/generic_red.cpp
            OBJECT_DEPENDS torch_headers.h
    )

    target_compile_options(
            ${template_name} BEFORE
            PRIVATE -include torch_headers.h
    )

    # We should include libtorch_python.so as an explicit include: https://github.com/pytorch/pytorch/issues/38122
    if (NOT APPLE AND NOT WIN32)
        target_link_libraries(
                ${template_name} PUBLIC
                ${PYTORCH_ROOT_DIR}/lib/libtorch_python.so
        )
    endif()

endif ()

# Ensure the shared lib look for the other .so in its own dir.
if (APPLE)
    set_target_properties(${template_name} PROPERTIES LINK_FLAGS "-Wl,-rpath,@loader_path/.")
else ()
    set_target_properties(${template_name} PROPERTIES LINK_FLAGS "-Wl,-rpath,$ORIGIN")
endif ()

# Write a log file to decypher keops dllname
include(../PyKeOpsLog.cmake)









