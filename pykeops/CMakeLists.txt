########################################################################################################################
#                                                       HEADERS                                                        #
########################################################################################################################

cmake_minimum_required(VERSION 3.10)

project(PyKeOps LANGUAGES CXX)

set(PYKEOPS_FOLDER ${CMAKE_CURRENT_SOURCE_DIR})

set(KEOPS_SRC ${PYKEOPS_FOLDER}/keops)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

include(${KEOPS_SRC}/cuda.cmake)

## Set Path to sources
set(SOURCE_FILES
        ${PYKEOPS_FOLDER}
        ${KEOPS_SRC}
        ${PROJECT_BINARY_DIR}
)

Include_Directories(${SOURCE_FILES})

include(${KEOPS_SRC}/headers.cmake)

# installation directory
set(BIN_DIR ${PROJECT_BINARY_DIR}/../)

########################################################################################################################
#                                                        KEOPS                                                         #
########################################################################################################################

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

if (USE_CUDA) # ----------------- create shared lib (cuda)

    add_library(
            keops${shared_obj_name} SHARED
            ${KEOPS_SRC}/core/link_autodiff.cu
    )

else () # ----------------- create shared lib (cpp)

    add_library(
            keops${shared_obj_name} SHARED
            ${KEOPS_SRC}/core/link_autodiff.cpp
    )

endif ()


target_compile_options(
        keops${shared_obj_name} BEFORE
        PRIVATE -include ${shared_obj_name}.h
)

# tell Cmake to explicitly add the dependency: keops is recompiled as soon as formula.h changes.
set_source_files_properties(
        ${KEOPS_SRC}/core/link_autodiff.cpp PROPERTIES
        OBJECT_DEPENDS ${shared_obj_name}.h
)

# set name
set_target_properties(
        keops${shared_obj_name} PROPERTIES
        LIBRARY_OUTPUT_NAME ${shared_obj_name}
        PREFIX ""
)

# skip the full RPATH for the build tree. We append $ORIGIN later
SET(CMAKE_SKIP_BUILD_RPATH FALSE)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
SET(CMAKE_INSTALL_RPATH "")
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)

########################################################################################################################
#                                                   INSTALLATION                                                       #
########################################################################################################################

add_custom_command(
        TARGET keops${shared_obj_name} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:keops${shared_obj_name}> ${BIN_DIR}
)

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



