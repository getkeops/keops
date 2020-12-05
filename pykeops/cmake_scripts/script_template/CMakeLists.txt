########################################################################################################################
#                                                       HEADERS                                                        #
########################################################################################################################

cmake_minimum_required(VERSION 3.10)

project(PyKeOps LANGUAGES CXX)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

set(PYKEOPS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../)

set(KEOPS_SOURCE_DIR ${PYKEOPS_SOURCE_DIR}/keops/)

add_definitions(-D__TYPE__=${__TYPE__})
add_definitions(-DC_CONTIGUOUS=${C_CONTIGUOUS})

## Set Path to sources
set(SOURCE_FILES
        ${PYKEOPS_SOURCE_DIR}
        ${PROJECT_BINARY_DIR}
		${KEOPS_SOURCE_DIR}
)

Include_Directories(${SOURCE_FILES})


# installation directory
set(BIN_DIR ${PROJECT_BINARY_DIR}/../)


#------------------------------------------------------------------------------------#
#------------------------------- COMPILATOR OPTS ------------------------------------#
#------------------------------------------------------------------------------------#

set(CMAKE_CXX_STANDARD 17)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  if(APPLE) 
	  # Apple built-in clang apparently does not support openmp...
	  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ferror-limit=2")
  else()
	  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_OPENMP -fopenmp -Wall -ferror-limit=2")
  endif()
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_OPENMP -fopenmp -Wall -Wno-unknown-pragmas -fmax-errors=2")
endif()

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")

if(APPLE)
  set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
  set(CMAKE_MACOSX_RPATH TRUE)
endif(APPLE)

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







########################################################################################################################
#                                                   INSTALLATION                                                       #
########################################################################################################################

#get_filename_component(PYKEOPS_SOURCE_DIR_ABS ${PYKEOPS_SOURCE_DIR} ABSOLUTE)

#add_custom_command(
#        TARGET ${template_name} POST_BUILD
#        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_OBJECTS:${template_name}> ${BIN_DIR}/${template_name}.o
#)







# Write a log file to decypher keops dllname
if (commandLine)
    string(TIMESTAMP TODAY "%Y/%m/%d")
    if (USE_CUDA)
        Set(COMPILER ${CMAKE_CUDA_COMPILER})
        Set(COMPILER_VERSION ${CMAKE_CUDA_COMPILER_VERSION})
    else ()
        Set(COMPILER ${CMAKE_CXX_COMPILER})
        Set(COMPILER_VERSION ${CMAKE_CXX_COMPILER_VERSION})
    endif ()
    file(APPEND ${PROJECT_BINARY_DIR}/../keops_hash.log
            "# ${shared_obj_name} compiled on ${TODAY} with ${COMPILER} (${COMPILER_VERSION}):\n\n ${commandLine}\n cmake --build . --target ${shared_obj_name} --  VERBOSE=1\n\n# ----------------------------------------------------------------------\n")
endif ()









