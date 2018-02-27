
# ----------------- test link
if (CUDA_FOUND)

    CUDA_add_executable(
        test_link_conv
        ${CMAKE_CURRENT_SOURCE_DIR}/test_link.cu
        OPTIONS --pre-include=formula.h
    )

    target_link_libraries(
        test_link_conv
        shared_obj
    )
endif()

# ----------------- equivalent code to compile_test_autodiff_cpp

add_executable(
    test_autodiff_cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_autodiff.cpp
    )

target_compile_options(
    test_autodiff_cpp BEFORE
    PRIVATE -include formula.h
)

# tell Cmake to explicitly add the dependency: shared_obj is recompiled as soon as formula.h changes.
set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/test_autodiff.cpp PROPERTIES
    OBJECT_DEPENDS formula.h
)



# ----------------- equivalent code to compile_test_factorized
if (CUDA_FOUND)

    CUDA_add_executable(
        test_factorized
        ${CMAKE_CURRENT_SOURCE_DIR}/test_factorized.cu
    )

endif()

# ----------------- equivalent code to compile_test_factorized_cpp
add_executable(
    test_factorized_cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_factorized.cpp
    )

target_compile_options(
    test_factorized_cpp BEFORE
    PRIVATE -include formula.h
)

# tell Cmake to explicitly add the dependency: shared_obj is recompiled as soon as formula.h changes.
set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/test_factorized.cpp PROPERTIES
    OBJECT_DEPENDS formula.h
)
