#------------------------------------------------------------------------------------#
#------------------------------- COMPILATOR OPTS ------------------------------------#
#------------------------------------------------------------------------------------#

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11 -Wall -ferror-limit=2")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11 -Wall -fmax-errors=2")
endif()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

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
    Set(__TYPE__ float CACHE STRING "Precision type of the computations (float or double)")
endif()
add_definitions(-D__TYPE__=${__TYPE__})

# - Declare the templates formula if not provided by the user
if(NOT DEFINED USENEWSYNTAX)
    Set(USENEWSYNTAX TRUE)
endif()

if(NOT USENEWSYNTAX)

    if(NOT FORMULA)
        Set(FORMULA "Sum_Reduction<Scal<Square<Scalprod<_X<3,4>,_Y<4,4>>>,GaussKernel<_P<0,1>,_X<1,3>,_Y<2,3>,_Y<5,3>>>>" CACHE STRING "Template formula to be instantiated")
    endif()
    unset(FORMULA_OBJ CACHE)

else()
    
    if(NOT FORMULA_OBJ)
        Set(VAR_ALIASES "auto x=Vi(1,3); auto y=Vj(2,3); auto u=Vi(3,4); auto v=Vj(4,4); auto b=Vj(5,3); auto p=Pm(0,1);")
        Set(FORMULA_OBJ "Sum_Reduction(Square((u|v))*Exp(-p*SqNorm2(x-y))*b,0)")
    endif()
    unset(FORMULA CACHE)

endif()

# We should generate a file to avoid parsing problem with shell: write the macros  in a file which will be included
configure_file(${CMAKE_CURRENT_LIST_DIR}/formula.h.in ${shared_obj_name}.h @ONLY)
