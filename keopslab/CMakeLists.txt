#------------------------------------------------------------------------------------#
#------------------------------------HEADERS-----------------------------------------#
#------------------------------------------------------------------------------------#

cmake_minimum_required(VERSION 3.10)

file(LOCK ${CMAKE_CURRENT_BINARY_DIR} DIRECTORY)
message(STATUS "Lock building directory: ${CMAKE_CURRENT_BINARY_DIR}")

project(KeOpsLab LANGUAGES CXX)

## Set Path to sources
set(KEOPS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../keops)

include(${KEOPS_SRC}/cuda.cmake)

set(SOURCE_FILES
    ${KEOPS_SRC}
    ${PROJECT_BINARY_DIR}
    )

Include_Directories(${SOURCE_FILES})

include(${KEOPS_SRC}/headers.cmake)

# installation directory
set(BIN_DIR ${PROJECT_BINARY_DIR}/../)


#------------------------------------------------------------------------------------#
#----------------------------------COMPILATIONS--------------------------------------#
#------------------------------------------------------------------------------------#

find_package(Matlab)

### - mexfiles : generate a working mexfile is painful with cmake. 
##   The problem is twofold : the cuda module do not define a 
##   CUDA_add_object. We then need to use Add_library instead.

if(Matlab_FOUND)

  # ----------------- get some env variables

  # this dummy flag is used to cast the input array of Matlab
  if(${__TYPE__} STREQUAL "double")
    add_definitions(-DUSE_DOUBLE)
  endif()

  if(NOT DEFINED shared_obj_name)
    Set(shared_obj_name mex_binding)
  endif()

  matlab_get_mex_suffix( # simply get the extension : almost surely mexa64
    ${Matlab_ROOT_DIR}
    mex_suffix
    )

  if(APPLE)
    set(rpat LDBUNDLE=\"-bundle -Wl,-rpath,@loader_path/.\")
  else()
    set(rpat LDFLAGS=\"-Wl,-rpath,\\\\\\$$ORIGIN\")
  endif()


  ########################################################################################################################
  #                                                        Generic                                                       #
  ########################################################################################################################

  if(USE_CUDA)

    add_library(
      keops${shared_obj_name} SHARED
      ${KEOPS_SRC}/core/link_autodiff.cu
    )

  else()

    # ----------------- create shared lib (cpp)

    add_library(
      keops${shared_obj_name} SHARED
      ${KEOPS_SRC}/core/link_autodiff.cpp
    )
  endif()


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
  set_target_properties(keops${shared_obj_name} PROPERTIES
                        # LIBRARY_OUTPUT_NAME  ${shared_obj_name}
                        PREFIX ""
                        )


  # ----------------- create mex files generic

  Add_library( # generate the string "g++ -c ..." but do not execute it
    mex_file_cpp OBJECT
    ${CMAKE_CURRENT_SOURCE_DIR}/generic/generic_red.cpp
    )

  target_compile_options(
    mex_file_cpp BEFORE
    PRIVATE -include ${shared_obj_name}.h -I${Matlab_INCLUDE_DIRS} -fPIC
  )

  add_custom_target(
    mex_cpp
    DEPENDS keops${shared_obj_name} mex_file_cpp # ensure obj file is created before using mex for linking
    COMMAND ${Matlab_ROOT_DIR}/bin/mex ${rpat} $<TARGET_FILE:keops${shared_obj_name}> $<TARGET_OBJECTS:mex_file_cpp> -output keops${shared_obj_name} # only since cmake 3.9
  )

  add_dependencies(
    mex_cpp
    keops${shared_obj_name}
  )

  set_target_properties( # pass needed options to add_custom_target()
    mex_cpp PROPERTIES
    PREFIX ""
    LINKER_LANGUAGE CXX
    EXCLUDE_FROM_ALL FALSE
    )


  # Installation step
  add_custom_command(
    TARGET keops${shared_obj_name} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:keops${shared_obj_name}> ${BIN_DIR}
  )

  add_custom_command(
    TARGET mex_cpp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy keops${shared_obj_name}.${mex_suffix} ${BIN_DIR}
  )

  # Write a log file to decypher keops dllname
  if(commandLine)
    string(TIMESTAMP TODAY "%Y/%m/%d")
    if(USE_CUDA)
      Set(COMPILER ${CMAKE_CUDA_COMPILER})
      Set(COMPILER_VERSION ${CMAKE_CUDA_COMPILER_VERSION})
    else()
      Set(COMPILER ${CMAKE_CXX_COMPILER})
      Set(COMPILER_VERSION ${CMAKE_CXX_COMPILER_VERSION})
    endif()
    file(APPEND ${PROJECT_BINARY_DIR}/../keops_hash.log
         "# ${shared_obj_name} compiled on ${TODAY} with ${COMPILER} (${COMPILER_VERSION}) and ${Matlab_ROOT_DIR}/bin/mex:\n\n ${commandLine}\n cmake --build . --target ${shared_obj_name} --  VERBOSE=1\n\n# ----------------------------------------------------------------------\n")
  endif()



  ########################################################################################################################
  #                                                        Specific                                                      #
  ########################################################################################################################

  if(USE_CUDA)

    # --------------------------- radial kernel conv
    add_library(
      radial_kernels_conv SHARED
      ${KEOPS_SRC}/specific/radial_kernels/cuda_conv.cu
    )

    add_library( # generate the string "g++ -c ..." but do not execute it
      mex_file_conv OBJECT
      ${CMAKE_CURRENT_SOURCE_DIR}/specific/convolutions/radial_kernel_conv.cpp
      )

    target_compile_options(
      mex_file_conv BEFORE
      PRIVATE -I${Matlab_INCLUDE_DIRS} -fPIC
    )

    add_custom_target(
      mex_conv
      DEPENDS radial_kernels_conv mex_file_conv # ensure obj file is created before using mex for linking
      COMMAND ${Matlab_ROOT_DIR}/bin/mex ${rpat} $<TARGET_FILE:radial_kernels_conv> $<TARGET_OBJECTS:mex_file_conv> -output conv # only since cmake 3.9
    )

    add_dependencies(
      mex_conv
      radial_kernels_conv
    )

    set_target_properties( # pass needed options to add_custom_target()
      mex_conv PROPERTIES
      PREFIX ""
      LINKER_LANGUAGE CXX
      EXCLUDE_FROM_ALL FALSE
      )


    # ----------------------------   fshape scp
    if(NOT KERNEL_GEOM OR (KERNEL_GEOM STREQUAL "gaussian"))
      SET(KERNEL_GEOM_TYPE 0)
    elseif(KERNEL_GEOM STREQUAL "cauchy")
      SET(KERNEL_GEOM_TYPE 1)
    else()
      message(FATAL_ERROR "Set KERNEL_GEOM type to gaussian or cauchy.")
    endif()
    add_definitions(-DKERNEL_GEOM_TYPE=${KERNEL_GEOM_TYPE})

    if(NOT KERNEL_SIG OR (KERNEL_SIG STREQUAL gaussian))
      SET(KERNEL_SIG_TYPE 0)
    elseif(KERNEL_SIG STREQUAL cauchy)
      SET(KERNEL_SIG_TYPE 1)
    else()
      message(FATAL_ERROR "Set KERNEL_SIG type to gaussian or cauchy.")
    endif()
    add_definitions(-DKERNEL_SIG_TYPE=${KERNEL_SIG_TYPE})

    if(NOT KERNEL_SPHERE OR (KERNEL_SPHERE STREQUAL gaussian_unoriented))
      SET(KERNEL_SPHERE_TYPE 0)
    elseif(KERNEL_SPHERE STREQUAL binet)
      SET(KERNEL_SPHERE_TYPE 1)
    elseif(KERNEL_SPHERE STREQUAL gaussian_oriented)
      SET(KERNEL_SPHERE_TYPE 2)
    elseif(KERNEL_SPHERE STREQUAL linear)
      SET(KERNEL_SPHERE_TYPE 3)
    else()
      message(FATAL_ERROR "Set KERNEL_SPHERE type to gaussian_unoriented, binet, gaussian_oriented or linear.")
    endif()
    add_definitions(-DKERNEL_SPHERE_TYPE=${KERNEL_SPHERE_TYPE})


    # ----------------- fshape_scp_dx
    foreach(ext_name "" "_dx" "_df" "_dxi")

      SET(mex_fshape_scp_name keops_fshape_scp${ext_name}_${KERNEL_GEOM}${KERNEL_SIG}${KERNEL_SPHERE})

      SET(name1 fshape_gpu${ext_name})
      add_library(
        ${name1} SHARED
        ${KEOPS_SRC}/specific/shape_distance/${name1}.cu
      )
      set_target_properties(${name1} PROPERTIES
                            LIBRARY_OUTPUT_NAME ${mex_fshape_scp_name}
                            PREFIX ""
                            )

      SET(name2 cudafshape${ext_name})
      Add_library( # generate the string "g++ -c ..." but do not execute it
        ${name2} OBJECT
        ${CMAKE_CURRENT_SOURCE_DIR}/specific/shape_distance/${name2}.cpp
        )
      target_compile_options(
        ${name2} BEFORE
        PRIVATE -I${Matlab_INCLUDE_DIRS} -fPIC
      )

      SET(name3 mex_fshape_scp${ext_name})
      if(NOT (${CMAKE_VERSION} VERSION_LESS 3.8.0))
        add_custom_target(
          ${name3}
          DEPENDS ${name1} ${name2}# ensure obj file is created before using mex for linking
          COMMAND ${Matlab_ROOT_DIR}/bin/mex ${rpat} $<TARGET_FILE:${name1}> $<TARGET_OBJECTS:${name2}> -output ${mex_fshape_scp_name} # only since cmake 3.9
        )
      else()
        add_custom_target(
          ${name3}
          DEPENDS ${name1} ${name2}# ensure obj file is created before using mex for linking
          COMMAND ${Matlab_ROOT_DIR}/bin/mex ${rpat} $<TARGET_FILE:${name1}> ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${name2}.dir/specific/shape_distance/cudafshape${ext_name}.cpp.o -output ${mex_fshape_scp_name}
        )
      endif()

      set_target_properties( # pass needed options to add_custom_target()
        ${name3} PROPERTIES
        PREFIX ""
        LINKER_LANGUAGE CXX
        EXCLUDE_FROM_ALL TRUE
        )

  # Installation step
  add_custom_command(
    TARGET ${name1} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${name1}> ${BIN_DIR}
  )

  add_custom_command(
    TARGET ${name3} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${mex_fshape_scp_name}.${mex_suffix} ${BIN_DIR}
  )
    endforeach()

  endif()
else()
  message(STATUS "Matlab not found. No mex file can be built.")
endif()

