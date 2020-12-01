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


#------------------------------------------------------------------------------------#
#----------------------------------- KeOps OPTS -------------------------------------#
#------------------------------------------------------------------------------------#

# Shared object name
if(NOT shared_obj_name)
  set(shared_obj_name keops)
endif()

message(STATUS "Using shared_obj_name: ${shared_obj_name}")

# - type for computation. The CACHE option enable to see it in ccmake.
if(NOT __TYPE__)
  Set(__TYPE__ float CACHE STRING "Precision type of the computations (__half, float or double)")
endif()
add_definitions(-D__TYPE__=${__TYPE__})

# - Choose if the multi-dimensional kernels are stored column or row wise 
if(NOT C_CONTIGUOUS)
  Set(C_CONTIGUOUS O CACHE STRING "Multi-dimensional kernels are stored column wise.")
endif()
add_definitions(-DC_CONTIGUOUS=${C_CONTIGUOUS})

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

# - Declare the templates formula if not provided by the user
if(NOT DEFINED USENEWSYNTAX)
  Set(USENEWSYNTAX 1)
endif()

