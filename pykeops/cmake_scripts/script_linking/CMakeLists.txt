########################################################################################################################
#                                                       HEADERS                                                        #
########################################################################################################################

cmake_minimum_required(VERSION 3.10)

project(PyKeOps LANGUAGES C CXX)



set(BIN_DIR ${PROJECT_BINARY_DIR}/../)





#------------------------------------------------------------------------------------#
#------------------------------- COMPILATOR OPTS ------------------------------------#
#------------------------------------------------------------------------------------#

set(CMAKE_CXX_STANDARD 17)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  if(APPLE) 
	  # Apple built-in clang apparently does not support openmp...
	  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -ferror-limit=2")
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









set(OBJS
  ${BIN_DIR}/${template_name}.o
  ${BIN_DIR}/${keops_formula_name}.o
)

add_library(${template_name} SHARED ${OBJS})

set_target_properties(${template_name} PROPERTIES LINKER_LANGUAGE CXX)

SET_SOURCE_FILES_PROPERTIES(
  ${OBJS}
  PROPERTIES
  EXTERNAL_OBJECT true
  GENERATED true
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



