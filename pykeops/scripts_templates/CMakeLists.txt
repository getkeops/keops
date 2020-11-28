########################################################################################################################
#                                                       HEADERS                                                        #
########################################################################################################################

cmake_minimum_required(VERSION 3.10)

project(PyKeOps LANGUAGES CXX)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

set(PYKEOPS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../)



## Set Path to sources
set(SOURCE_FILES
        ${PYKEOPS_SOURCE_DIR}
        ${PROJECT_BINARY_DIR}
)

Include_Directories(${SOURCE_FILES})


# installation directory
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

########################################################################################################################
#                                                       PYTORCH                                                        #
########################################################################################################################

if (NOT DEFINED PYTHON_LANG)
    Set(PYTHON_LANG numpy)
endif ()

if (${PYTHON_LANG} STREQUAL "torch")

    # The  following lines could replace the included files below. Unfortunately, the TorchConfig.cmake import too many
    # library (caffe2 ...) which is not relevent in our case.
    #set(CMAKE_PREFIX_PATH ${PYTORCH_ROOT_DIR})
    #find_package(Torch REQUIRED)
    #add_definitions(${TORCH_CXX_FLAGS})

    Include_Directories(
            ${PYTORCH_ROOT_DIR}/include/
            ${PYTORCH_ROOT_DIR}/include/torch/csrc/api/include/
    )

    # fix for pytorch: https://discuss.pytorch.org/t/pytorch-0-4-1-undefined-symbol-at-import-of-a-cpp-extension/24420
    # and https://stackoverflow.com/questions/33394934/converting-std-cxx11string-to-stdstring
    if(NOT DEFINED _GLIBCXX_USE_CXX11_ABI)
        Set(_GLIBCXX_USE_CXX11_ABI 0)  # set default value to False...
    endif()
    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=${_GLIBCXX_USE_CXX11_ABI})

    # We should generate a file to avoid parsing problem with shell: write the macros in a file which will be included
    configure_file(${PYKEOPS_SOURCE_DIR}/torch_headers.h.in ${CMAKE_CURRENT_BINARY_DIR}/torch_headers.h @ONLY)
endif()


########################################################################################################################
#                                                     PYBIND11                                                         #
########################################################################################################################

add_subdirectory(${PYKEOPS_SOURCE_DIR}/pybind11 ${CMAKE_CURRENT_BINARY_DIR}/pybind11)  #find_package(pybind11  REQUIRED)
add_definitions(-DMODULE_NAME=${shared_obj_name})

pybind11_add_module(${shared_obj_name}
        ${PYKEOPS_SOURCE_DIR}/${PYTHON_LANG}/generic/generic_red.cpp
)

if(${PYTHON_LANG} STREQUAL "torch")

    set_source_files_properties(
            ${PYKEOPS_SOURCE_DIR}/${PYTHON_LANG}/generic/generic_red.cpp
            OBJECT_DEPENDS torch_headers.h
    )

    target_compile_options(
            ${shared_obj_name} BEFORE
            PRIVATE -include torch_headers.h
    )

    # We should include libtorch_python.so as an explicit include: https://github.com/pytorch/pytorch/issues/38122
    if (NOT APPLE AND NOT WIN32)
        target_link_libraries(
                ${shared_obj_name} PUBLIC
                ${PYTORCH_ROOT_DIR}/lib/libtorch_python.so
        )
    endif()

endif ()

#SET_TARGET_PROPERTIES(${shared_obj_name} PROPERTIES RULE_LAUNCH_LINK
#    "${CMAKE_CURRENT_SOURCE_DIR}/cpo ${BIN_DIR} <OBJECTS> --"
#)

# Ensure the shared lib look for the other .so in its own dir.
if (APPLE)
    set_target_properties(${shared_obj_name} PROPERTIES LINK_FLAGS "-Wl,-rpath,@loader_path/.")
else ()
    set_target_properties(${shared_obj_name} PROPERTIES LINK_FLAGS "-Wl,-rpath,$ORIGIN")
endif ()


########################################################################################################################
#                                                   INSTALLATION                                                       #
########################################################################################################################

add_custom_command(
        TARGET ${shared_obj_name} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${shared_obj_name}> ${BIN_DIR}
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











########################################################################################################################
#                                                        Specific                                                      #
########################################################################################################################

if (USE_CUDA)

    # --------------------------------------------------- CONV ---------------------------------------------------------
    add_library(
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
    if (APPLE)
        set_target_properties(radial_kernel_conv PROPERTIES LINK_FLAGS "-Wl,-rpath,@loader_path/.")
    else ()
        set_target_properties(radial_kernel_conv PROPERTIES LINK_FLAGS "-Wl,-rpath,$ORIGIN")
    endif ()

    # Installation step
    add_custom_command(
            TARGET radial_kernel_conv POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:radial_kernel_conv> ${BIN_DIR}
    )
    add_custom_command(
            TARGET radial_kernel_conv_cuda POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:radial_kernel_conv_cuda> ${BIN_DIR}
    )


    # ------------------------------------------------ GRAD1CONV -------------------------------------------------------

    add_library(
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
    if (APPLE)
        set_target_properties(radial_kernel_grad1conv PROPERTIES LINK_FLAGS "-Wl,-rpath,@loader_path/.")
    else ()
        set_target_properties(radial_kernel_grad1conv PROPERTIES LINK_FLAGS "-Wl,-rpath,$ORIGIN")
    endif ()

    # Installation step
    add_custom_command(
            TARGET radial_kernel_grad1conv POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:radial_kernel_grad1conv> ${BIN_DIR}
    )
    add_custom_command(
            TARGET radial_kernel_grad1conv_cuda POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:radial_kernel_grad1conv_cuda> ${BIN_DIR}
    )


    # --------------------------------------------------- FSHAPE SCP  --------------------------------------------------

    if (NOT KERNEL_GEOM OR (KERNEL_GEOM STREQUAL "gaussian"))
        SET(KERNEL_GEOM "gaussian")
        SET(KERNEL_GEOM_TYPE 0)
    elseif (KERNEL_GEOM STREQUAL "cauchy")
        SET(KERNEL_GEOM_TYPE 1)
    else ()
        message(FATAL_ERROR "Set KERNEL_GEOM type to gaussian or cauchy.")
    endif ()
    add_definitions(-DKERNEL_GEOM_TYPE=${KERNEL_GEOM_TYPE})

    if (NOT KERNEL_SIG OR (KERNEL_SIG STREQUAL gaussian))
        SET(KERNEL_SIG "gaussian")
        SET(KERNEL_SIG_TYPE 0)
    elseif (KERNEL_SIG STREQUAL cauchy)
        SET(KERNEL_SIG_TYPE 1)
    else ()
        message(FATAL_ERROR "Set KERNEL_SIG type to gaussian or cauchy.")
    endif ()
    add_definitions(-DKERNEL_SIG_TYPE=${KERNEL_SIG_TYPE})

    if (NOT KERNEL_SPHERE OR (KERNEL_SPHERE STREQUAL gaussian_unoriented))
        SET(KERNEL_SPHERE "gaussian_unoriented")
        SET(KERNEL_SPHERE_TYPE 0)
    elseif (KERNEL_SPHERE STREQUAL binet)
        SET(KERNEL_SPHERE_TYPE 1)
    elseif (KERNEL_SPHERE STREQUAL gaussian_oriented)
        SET(KERNEL_SPHERE_TYPE 2)
    elseif (KERNEL_SPHERE STREQUAL linear)
        SET(KERNEL_SPHERE_TYPE 3)
    else ()
        message(FATAL_ERROR "Set KERNEL_SPHERE type to gaussian_unoriented, binet, gaussian_oriented or linear.")
    endif ()
    add_definitions(-DKERNEL_SPHERE_TYPE=${KERNEL_SPHERE_TYPE})

    #foreach(ext_name "" "_dx" "_df" "_dxi")
    foreach (ext_name "")

        SET(fshape_scp_name fshape_scp${ext_name}_${KERNEL_GEOM}${KERNEL_SIG}${KERNEL_SPHERE}_${__TYPE__})

        SET(name1 fshape_gpu${ext_name})

        add_library(
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
        if (APPLE)
            set_target_properties(${fshape_scp_name} PROPERTIES LINK_FLAGS "-Wl,-rpath,@loader_path/.")
        else ()
            set_target_properties(${fshape_scp_name} PROPERTIES LINK_FLAGS "-Wl,-rpath,$ORIGIN")
        endif ()

        # Installation step
        add_custom_command(
                TARGET ${fshape_scp_name} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${fshape_scp_name}> ${BIN_DIR}
        )
        add_custom_command(
                TARGET ${name1} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${name1}> ${BIN_DIR}
        )


    endforeach ()

endif ()

