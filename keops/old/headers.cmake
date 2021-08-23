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

# - Choose if the multi-dimensional kernels are stored column or row wise
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

# - recover the declared positions of the variables in a Formula call
string(REGEX REPLACE " " "" FORMULA_NOSPACE ${FORMULA_OBJ} ${VAR_ALIASES})
string(REGEX MATCHALL "(Pm|V(ar|[ij]))\\(([0-9]+)" ARGS_LIST ${FORMULA_NOSPACE})
string(REGEX REPLACE "(Pm|V(ar|[ij]))\\(" ";" ARGS_POS_LIST ${ARGS_LIST})

# - Count the number of variable in the formula: it depends on alias(es), explicit Var(pos, dim, type) declation(s) and GradFromPos(_WithSavedForward) operators.
list(REMOVE_DUPLICATES ARGS_POS_LIST)
list(LENGTH ARGS_POS_LIST TMP)
MATH(EXPR NARGS "${TMP} - 1")

string(REGEX MATCHALL "GradFrom(Pos|Ind)\\(" GFP_LIST ${FORMULA_NOSPACE})
if(GFP_LIST)
  string(REGEX REPLACE "GradFrom(Pos|Ind)\\(" "a;" GFP_LIST_2 ${GFP_LIST})
  list(LENGTH GFP_LIST_2 MM)
  set(TMP ${NARGS})
  MATH(EXPR NARGS " ${MM} - 1 + ${TMP}")  # Add implicitely 1 variable
endif()

string(REGEX MATCHALL "GradFrom(Pos|Ind)_WithSavedForward\\(" GFP_LIST ${FORMULA_NOSPACE})
if(GFP_LIST)
  string(REGEX REPLACE "GradFrom(Pos|Ind)\\(" "a;" GFP_LIST_2 ${GFP_LIST})
  list(LENGTH GFP_LIST_2 MM)
  set(TMP ${NARGS})
  MATH(EXPR NARGS " ${MM} + ${TMP}")    # Add implicitely 2 variables
endif()

# - recover the position of the first I variable:
string(REGEX MATCH "Vi\\(([0-9]+)" ARGI_FIRST ${FORMULA_NOSPACE})

if(ARGI_FIRST)
  set(POS_FIRST_ARGI ${CMAKE_MATCH_1})
endif()

if(NOT ARGI_FIRST)
  string(REGEX MATCH "Var\\(([0-9]+),[0-9]+,0" ARGI_FIRST ${FORMULA_NOSPACE})
  if(ARGI_FIRST)
    set(POS_FIRST_ARGI ${CMAKE_MATCH_1})
  endif()
endif()

if(NOT ARGI_FIRST)
  set(POS_FIRST_ARGI "-1")
  message(STATUS "No i variables detected")
else()
  message(STATUS "First i variables detected is ${POS_FIRST_ARGI}")
endif()

# - recover the position of the first J variable:
string(REGEX MATCH "Vj\\(([0-9]+)" ARGJ_FIRST ${FORMULA_NOSPACE})
if(ARGJ_FIRST)
  set(POS_FIRST_ARGJ ${CMAKE_MATCH_1})
endif()

if(NOT ARGJ_FIRST)
  string(REGEX MATCH "Var\\(([0-9]+),[0-9]+,1" ARGJ_FIRST ${FORMULA_NOSPACE})
  if(ARGJ_FIRST)
    set(POS_FIRST_ARGJ ${CMAKE_MATCH_1})
  endif()
endif()

if(NOT ARGJ_FIRST)
  set(POS_FIRST_ARGJ "-1")
  message(STATUS "No j variables detected.")
else()
  message(STATUS "First j variables detected is ${POS_FIRST_ARGJ}")
endif()


message(STATUS "Compiled formula is ${FORMULA_OBJ}; ${VAR_ALIASES} where the number of args is ${NARGS}.")
# We should generate a file to avoid parsing problem with shell: write the macros  in a file which will be included
configure_file(${CMAKE_CURRENT_LIST_DIR}/formula.h.in ${shared_obj_name}.h @ONLY)

