if(NOT (APPLE))
  add_executable(
    test_tensordot_c14
    ${CMAKE_CURRENT_SOURCE_DIR}/test_tensordot_c14.cpp
  )
  SET_PROPERTY(
    TARGET test_tensordot_c14
    PROPERTY CXX_STANDARD 14
  )


  add_executable(
    test_tensordot
    ${CMAKE_CURRENT_SOURCE_DIR}/test_tensordot.cpp
  )

  add_executable(
          notao_cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/notao.cpp
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

  add_executable(
    test_specific
    ${CMAKE_CURRENT_SOURCE_DIR}/test_specific.cu
  )

  add_executable(
    test_factorized
    ${CMAKE_CURRENT_SOURCE_DIR}/test_factorized.cu
  )

endif()

