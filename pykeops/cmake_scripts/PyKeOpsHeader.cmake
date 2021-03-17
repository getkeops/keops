#------------------------------------------------------------------------------------#
#------------------------------- COMPILATOR OPTS ------------------------------------#
#------------------------------------------------------------------------------------#

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# N.B. (Joan, 2021/03/17) I replaced the set(CMAKE_CXX_FLAGS ...) commands by
# add_definitions(...) because otherwise it is not teken into account when
# compiling with nvcc ; I don't know why. 
# The set(CMAKE_CXX_FLAGS_RELEASE ...) and set(CMAKE_CXX_FLAGS_DEBUG(...)
# are probably not recognised either...

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  if(APPLE)
	  # Apple built-in clang apparently does not support openmp...
	  # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ferror-limit=2")
      add_definitions("-ferror-limit=2")
  else()
	  # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_OPENMP -fopenmp -Wall -ferror-limit=2")
      add_definitions("-DUSE_OPENMP -fopenmp -Wall -ferror-limit=2")
  endif()
else()
  # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_OPENMP -fopenmp -Wall -Wno-unknown-pragmas -fmax-errors=2")
  add_definitions("-DUSE_OPENMP -fopenmp -Wall -Wno-unknown-pragmas -fmax-errors=2")
endif()

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")

if(APPLE)
  set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
  set(CMAKE_MACOSX_RPATH TRUE)
endif(APPLE)
