if(NOT (APPLE))
  add_executable(
    test_tensordot_c14
    ${CMAKE_CURRENT_SOURCE_DIR}/test_tensordot_c14.cpp
  )

  add_executable(
    test_tensordot
    ${CMAKE_CURRENT_SOURCE_DIR}/test_tensordot.cpp
  )
  SET_PROPERTY(TARGET test_tensordot
               PROPERTY CXX_STANDARD 17
               )
endif()

add_executable(
  test_factorized_cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/test_factorized.cpp
)

add_executable(
  test_sumgauss_cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/test_sumgauss.cpp
)

add_executable(
  test_specific_cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/test_specific.cpp
)

if(USE_CUDA)

  CUDA_add_executable(
    test_specific
    ${CMAKE_CURRENT_SOURCE_DIR}/test_specific.cu
  )

  CUDA_add_executable(
    test_factorized
    ${CMAKE_CURRENT_SOURCE_DIR}/test_factorized.cu
  )

endif()

