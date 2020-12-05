########################################################################################################################
#                                                       HEADERS                                                        #
########################################################################################################################

cmake_minimum_required(VERSION 3.10)

project(PyKeOps LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)

set(BIN_DIR ${PROJECT_BINARY_DIR}/../)

include(../PyKeOpsHeader.cmake)

########################################################################################################################
#                                                       LINKING                                                        #
########################################################################################################################

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



