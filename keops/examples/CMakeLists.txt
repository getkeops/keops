add_executable(
  test_formula_without_j_cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/test_formula_without_j.cpp
)

add_executable(
  test_tensordot_formula_cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/test_tensordot_formula.cpp
)

add_executable(
  test_simple_cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/test_simple.cpp
)

add_executable(
  test_autodiff_cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/test_autodiff.cpp
)

add_executable(
  test_reductions_cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/test_reductions.cpp
)

if(USE_CUDA)

  CUDA_add_executable(
    test_formula_without_j
    ${CMAKE_CURRENT_SOURCE_DIR}/test_formula_without_j.cu
  )

  CUDA_add_executable(
    test_gpytorch
    ${CMAKE_CURRENT_SOURCE_DIR}/test_gpytorch.cu
  )

  CUDA_add_executable(
    test_tensordot_formula
    ${CMAKE_CURRENT_SOURCE_DIR}/test_tensordot_formula.cu
  )

  CUDA_add_executable(
    test_simple
    ${CMAKE_CURRENT_SOURCE_DIR}/test_simple.cu
  )

  CUDA_add_executable(
    test_autodiff
    ${CMAKE_CURRENT_SOURCE_DIR}/test_autodiff.cu
  )

  CUDA_add_executable(
    test_fromdevice
    ${CMAKE_CURRENT_SOURCE_DIR}/test_fromdevice.cu
  )

  CUDA_add_executable(
    test_reductions
    ${CMAKE_CURRENT_SOURCE_DIR}/test_reductions.cu
  )

endif()

