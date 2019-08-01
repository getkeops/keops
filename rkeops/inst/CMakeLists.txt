#------------------------------------------------------------------------------------#
#------------------------------------HEADERS-----------------------------------------#
#------------------------------------------------------------------------------------#

cmake_minimum_required(VERSION 3.10)

file(LOCK ${CMAKE_CURRENT_BINARY_DIR} DIRECTORY)
message(STATUS "Lock building directory: ${CMAKE_CURRENT_BINARY_DIR}")

project(RKeOps LANGUAGES CXX)

## Set Path to sources
# FIXME
set(KEOPS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../keops)

include(${KEOPS_SRC}/cuda.cmake)

set(SOURCE_FILES
    ${KEOPS_SRC}
    ${PROJECT_BINARY_DIR}
)

Include_Directories(${SOURCE_FILES})

include(${KEOPS_SRC}/headers.cmake)


#------------------------------------------------------------------------------------#
#----------------------------------COMPILATIONS--------------------------------------#
#------------------------------------------------------------------------------------#

# FIXME