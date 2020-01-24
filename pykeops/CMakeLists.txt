#------------------------------------------------------------------------------------#
#------------------------------------HEADERS-----------------------------------------#
#------------------------------------------------------------------------------------#

cmake_minimum_required(VERSION 3.10)

project(PyKeOps LANGUAGES CXX)

set(KEOPS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/keops)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

include(${KEOPS_SRC}/cuda.cmake)

## Set Path to sources
set(SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${KEOPS_SRC}
    ${PROJECT_BINARY_DIR}
    )

if(PYTORCH_INCLUDE_DIR)
  list(APPEND SOURCE_FILES ${PYTORCH_INCLUDE_DIR})
endif()

Include_Directories(${SOURCE_FILES})

include(${KEOPS_SRC}/headers.cmake)

# installation directory
set(BIN_DIR ${PROJECT_BINARY_DIR}/../)


########################################################################################################################
#                                                     PYBIND11                                                         #
########################################################################################################################

add_subdirectory(pybind11)
add_definitions(-DMODULE_NAME=${shared_obj_name})
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0) #  fix for pytorch: https://discuss.pytorch.org/t/pytorch-0-4-1-undefined-symbol-at-import-of-a-cpp-extension/24420 and https://stackoverflow.com/questions/33394934/converting-std-cxx11string-to-stdstring
if(NOT PYTHON_LANG)
  Set(PYTHON_LANG numpy)
endif()

# We should generate a file to avoid parsing problem with shell: write the macros  in a file which will be included
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/torch_headers.h.in ${CMAKE_CURRENT_BINARY_DIR}/torch_headers.h @ONLY)


########################################################################################################################
#                                                        Generic                                                       #
########################################################################################################################

# this dummy flag is used in the bindings
if(${__TYPE__} STREQUAL "double")
  add_definitions(-DUSE_DOUBLE=1)
else()
  add_definitions(-DUSE_DOUBLE=0)
endif()


# ----------------- create shared lib (cuda)
if(USE_CUDA)

  CUDA_add_library(
    keops${shared_obj_name} SHARED
    ${KEOPS_SRC}/core/link_autodiff.cu
    OPTIONS --pre-include=${shared_obj_name}.h
  )

else()
  # ----------------- create shared lib (cpp)

  add_library(
    keops${shared_obj_name} SHARED
    ${KEOPS_SRC}/core/link_autodiff.cpp
  )

  target_compile_options(
    keops${shared_obj_name} BEFORE
    PRIVATE -include ${shared_obj_name}.h
  )

  # tell Cmake to explicitly add the dependency: keops is recompiled as soon as formula.h changes.
  set_source_files_properties(
    ${KEOPS_SRC}/core/link_autodiff.cpp PROPERTIES
    OBJECT_DEPENDS ${shared_obj_name}.h
  )

endif()

# skip the full RPATH for the build tree. We append $ORIGIN later
SET(CMAKE_SKIP_BUILD_RPATH FALSE)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
SET(CMAKE_INSTALL_RPATH "")
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)

# set name
set_target_properties(keops${shared_obj_name} PROPERTIES
                      LIBRARY_OUTPUT_NAME ${shared_obj_name}
                      PREFIX ""
                      )

# -------------------- Pybind11
pybind11_add_module(${shared_obj_name}
                    ${CMAKE_CURRENT_SOURCE_DIR}/${PYTHON_LANG}/generic/generic_red.cpp
                    )

if(${PYTHON_LANG} STREQUAL "torch")
  set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/${PYTHON_LANG}/generic/generic_red.cpp
    OBJECT_DEPENDS torch_headers.h
  )

  target_compile_options(
    ${shared_obj_name} BEFORE
    PRIVATE -include torch_headers.h
  )
else() # assuming numpy

  target_compile_options(
    ${shared_obj_name} BEFORE
    PRIVATE -include ${shared_obj_name}.h
  )

endif()

# Ensure the shared lib look for the other .so in its own dir.
if(APPLE)
  set_target_properties(${shared_obj_name} PROPERTIES LINK_FLAGS "-Wl,-rpath,@loader_path/.")
else()
  set_target_properties(${shared_obj_name} PROPERTIES LINK_FLAGS "-Wl,-rpath,$ORIGIN")
endif()

target_link_libraries(
  ${shared_obj_name} PUBLIC
  keops${shared_obj_name}
)

# Installation step
add_custom_command(
  TARGET ${shared_obj_name} POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${shared_obj_name}> ${BIN_DIR}
)
add_custom_command(
  TARGET keops${shared_obj_name} POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:keops${shared_obj_name}> ${BIN_DIR}
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
       "# ${shared_obj_name} compiled on ${TODAY} with ${COMPILER} (${COMPILER_VERSION}):\n\n ${commandLine}\n cmake --build . --target ${shared_obj_name} --  VERBOSE=1\n\n# ----------------------------------------------------------------------\n")
endif()


########################################################################################################################
#                                                        Specific                                                      #
########################################################################################################################

if(USE_CUDA)

  # --------------------------------------------------- CONV -------------------------------------------------------------
  CUDA_add_library(
    radial_kernel_conv_cuda SHARED
    ${KEOPS_SRC}/specific/radial_kernels/cuda_conv.cu
  )

  pybind11_add_module(radial_kernel_conv
                      ${CMAKE_CURRENT_SOURCE_DIR}/numpy/convolutions/radial_kernel_conv.cpp
                      )

  target_compile_options(
    radial_kernel_conv BEFORE
    PRIVATE -include ${shared_obj_name}.h
  )

  target_link_libraries(
    radial_kernel_conv PUBLIC
    radial_kernel_conv_cuda
  )

  # Ensure the shared lib look for the other .so in its own dir.
  if(APPLE)
    set_target_properties(radial_kernel_conv PROPERTIES LINK_FLAGS "-Wl,-rpath,@loader_path/.")
  else()
    set_target_properties(radial_kernel_conv PROPERTIES LINK_FLAGS "-Wl,-rpath,$ORIGIN")
  endif()

  # Installation step
  add_custom_command(
    TARGET radial_kernel_conv POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:radial_kernel_conv> ${BIN_DIR}
  )
  add_custom_command(
    TARGET radial_kernel_conv_cuda POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:radial_kernel_conv_cuda> ${BIN_DIR}
  )


  # ------------------------------------------------ GRAD1CONV -----------------------------------------------------------

  CUDA_add_library(
    radial_kernel_grad1conv_cuda SHARED
    ${KEOPS_SRC}/specific/radial_kernels/cuda_grad1conv.cu
  )

  pybind11_add_module(radial_kernel_grad1conv
                      ${CMAKE_CURRENT_SOURCE_DIR}/numpy/convolutions/radial_kernel_grad1conv.cpp
                      )

  target_compile_options(
    radial_kernel_grad1conv BEFORE
    PRIVATE -include ${shared_obj_name}.h
  )

  target_link_libraries(
    radial_kernel_grad1conv PUBLIC
    radial_kernel_grad1conv_cuda
  )

  # Ensure the shared lib look for the other .so in its own dir.
  if(APPLE)
    set_target_properties(radial_kernel_grad1conv PROPERTIES LINK_FLAGS "-Wl,-rpath,@loader_path/.")
  else()
    set_target_properties(radial_kernel_grad1conv PROPERTIES LINK_FLAGS "-Wl,-rpath,$ORIGIN")
  endif()

  # Installation step
  add_custom_command(
    TARGET radial_kernel_grad1conv POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:radial_kernel_grad1conv> ${BIN_DIR}
  )
  add_custom_command(
    TARGET radial_kernel_grad1conv_cuda POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:radial_kernel_grad1conv_cuda> ${BIN_DIR}
  )


  # --------------------------------------------------- FSHAPE SCP  ------------------------------------------------------

  if(NOT KERNEL_GEOM OR (KERNEL_GEOM STREQUAL "gaussian"))
    SET(KERNEL_GEOM "gaussian")
    SET(KERNEL_GEOM_TYPE 0)
  elseif(KERNEL_GEOM STREQUAL "cauchy")
    SET(KERNEL_GEOM_TYPE 1)
  else()
    message(FATAL_ERROR "Set KERNEL_GEOM type to gaussian or cauchy.")
  endif()
  add_definitions(-DKERNEL_GEOM_TYPE=${KERNEL_GEOM_TYPE})

  if(NOT KERNEL_SIG OR (KERNEL_SIG STREQUAL gaussian))
    SET(KERNEL_SIG "gaussian")
    SET(KERNEL_SIG_TYPE 0)
  elseif(KERNEL_SIG STREQUAL cauchy)
    SET(KERNEL_SIG_TYPE 1)
  else()
    message(FATAL_ERROR "Set KERNEL_SIG type to gaussian or cauchy.")
  endif()
  add_definitions(-DKERNEL_SIG_TYPE=${KERNEL_SIG_TYPE})

  if(NOT KERNEL_SPHERE OR (KERNEL_SPHERE STREQUAL gaussian_unoriented))
    SET(KERNEL_SPHERE "gaussian_unoriented")
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

  #foreach(ext_name "" "_dx" "_df" "_dxi")
  foreach(ext_name "")

    SET(fshape_scp_name fshape_scp${ext_name}_${KERNEL_GEOM}${KERNEL_SIG}${KERNEL_SPHERE}_${__TYPE__})

    SET(name1 fshape_gpu${ext_name})

    CUDA_add_library(
      ${name1} SHARED
      ${KEOPS_SRC}/specific/shape_distance/${name1}.cu
    )

    set_target_properties(${name1} PROPERTIES
                          LIBRARY_OUTPUT_NAME ${fshape_scp_name}
                          PREFIX ""
                          )

    SET(name2 fshape_scp${ext_name})
    add_definitions(-DMODULE_NAME_FSHAPE_SCP=${fshape_scp_name})

    pybind11_add_module(${fshape_scp_name}
                        ${CMAKE_CURRENT_SOURCE_DIR}/numpy/shape_distance/${name2}.cpp
                        )

    target_compile_options(
      ${fshape_scp_name} BEFORE
      PRIVATE -include ${shared_obj_name}.h
    )

    target_link_libraries(
      ${fshape_scp_name} PUBLIC
      ${name1}
    )
    # Ensure the shared lib look for the other .so in its own dir.
    if(APPLE)
      set_target_properties(${fshape_scp_name} PROPERTIES LINK_FLAGS "-Wl,-rpath,@loader_path/.")
    else()
      set_target_properties(${fshape_scp_name} PROPERTIES LINK_FLAGS "-Wl,-rpath,$ORIGIN")
    endif()

    # Installation step
    add_custom_command(
      TARGET ${fshape_scp_name} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${fshape_scp_name}> ${BIN_DIR}
    )
    add_custom_command(
      TARGET ${name1} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${name1}> ${BIN_DIR}
    )


  endforeach()

endif()

