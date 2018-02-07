
# ----------------- equivalent code to compile_test_autodiff_cpp

add_executable(
    test_autodiff_cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_autodiff.cpp
    )

target_compile_options(
    test_autodiff_cpp BEFORE
    PRIVATE -include template.h
)




# ----------------- equivalent code to compile_test_factorized_cpp

add_executable(
    test_factorized_cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_factorized.cpp
    )

target_compile_options(
    test_factorized_cpp BEFORE
    PRIVATE -include template.h
)

