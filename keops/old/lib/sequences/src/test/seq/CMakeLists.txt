cmake_minimum_required (VERSION 3.3.0 FATAL_ERROR)

set (test_sources
  accumulate.cpp
  at_index.cpp
  concatenate.cpp
  contains.cpp
  contrib_index_of_seq.cpp
  contrib_permutate.cpp
  contrib_sort_index.cpp
  difference.cpp
  exclusive_scan.cpp
  head.cpp
  inclusive_scan.cpp
  index_of.cpp
  integer_sequence.cpp
  is_all.cpp
  is_any.cpp
  make_integer_range.cpp
  make_integer_sequence.cpp
  map.cpp
  max.cpp
  min.cpp
  minus.cpp
  multiplies.cpp
  partial_accumulate.cpp
  partial_sum.cpp
  plus.cpp
  prod.cpp
  reverse.cpp
  scale.cpp
  select.cpp
  sort.cpp
  sum.cpp
  tail.cpp
  tester.cpp
)

# file (GLOB ...) is used to validate the above list of test_sources
file (GLOB glob_test_sources RELATIVE ${CMAKE_CURRENT_LIST_DIR} *.cpp)

foreach (testsourcefile ${test_sources})
  if (${testsourcefile} IN_LIST glob_test_sources)
    list (REMOVE_ITEM glob_test_sources ${testsourcefile})
  else ()
    message (SEND_ERROR "File ${testsourcefile} is missing from src/test/seq")
  endif ()

  get_filename_component (exename ${testsourcefile} NAME_WE)
  set (exename "tao-sequences-test-${exename}")
  add_executable (${exename} ${testsourcefile})
  target_link_libraries (${exename} PRIVATE taocpp::sequences)
  set_target_properties (${exename} PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
  )
  if (MSVC)
    target_compile_options (${exename} PRIVATE /W4 /WX /utf-8)
  else ()
    target_compile_options (${exename} PRIVATE -pedantic -Wall -Wextra -Wshadow -Werror)
  endif ()
  add_test (NAME ${exename} WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../.. COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${exename})
endforeach (testsourcefile)

if (glob_test_sources)
  foreach (ignored_source_file ${glob_test_sources})
    message (SEND_ERROR "File ${ignored_source_file} in src/test/seq is ignored")
  endforeach (ignored_source_file)
endif ()
