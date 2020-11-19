enable_testing()

find_package(Threads)

function(add_test_helper test_name source_file link_lib)
  add_executable(${test_name} ${source_file})
  if(${link_lib} STREQUAL "")
    target_link_libraries(${test_name} gtest gtest_main)
  else()
    target_link_libraries(${test_name} ${link_lib} gtest gtest_main)
  endif()
  add_test(${test_name} ${test_name})
endfunction()

# Multi thread compilation test
set(empty "")

add_test_helper(unit_test_compile gtest_compile.cpp empty)

add_test_helper(unit_test_tensordot gtest_tensordot.cpp empty)

if (USE_CUDA)
  add_test_helper(unit_test_grad1conv ${CMAKE_CURRENT_SOURCE_DIR}/gtest_grad1conv.cu radial_kernels_grad1conv)

  add_test_helper(unit_test_conv ${CMAKE_CURRENT_SOURCE_DIR}/gtest_conv.cpp ${shared_obj_name})
  
  add_test_helper(unit_test_tensordot_cuda gtest_tensordot.cu empty)
endif()
