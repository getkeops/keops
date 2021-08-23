cmake_minimum_required (VERSION 3.2.0 FATAL_ERROR)

project (taocpp-sequences VERSION 2.0.1 LANGUAGES CXX)

# installation directories
set (TAOCPP_SEQUENCES_INSTALL_INCLUDE_DIR "include" CACHE STRING "The installation include directory")
set (TAOCPP_SEQUENCES_INSTALL_DOC_DIR "share/doc/tao/seq" CACHE STRING "The installation doc directory")
set (TAOCPP_SEQUENCES_INSTALL_CMAKE_DIR "share/taocpp-sequences/cmake" CACHE STRING "The installation cmake directory")

# define a header-only library
add_library (sequences INTERFACE)
add_library (taocpp::sequences ALIAS sequences)
target_include_directories (sequences INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${TAOCPP_SEQUENCES_INSTALL_INCLUDE_DIR}>
)

# features used by taocpp/sequences
target_compile_features (sequences INTERFACE
  cxx_alias_templates
  cxx_constexpr
  cxx_decltype
  cxx_noexcept
  cxx_static_assert
  cxx_variadic_templates
)

# testing
enable_testing ()
option (TAOCPP_SEQUENCES_BUILD_TESTS "Build test programs" ON)
if (TAOCPP_SEQUENCES_BUILD_TESTS)
  add_subdirectory (src/test/seq)
endif ()

# install and export target
install (TARGETS sequences EXPORT sequences-targets)

install (EXPORT sequences-targets
  FILE taocpp-sequences-config.cmake
  NAMESPACE taocpp::
  DESTINATION ${TAOCPP_SEQUENCES_INSTALL_CMAKE_DIR}
)

install (DIRECTORY include/ DESTINATION ${TAOCPP_SEQUENCES_INSTALL_INCLUDE_DIR})
install (FILES LICENSE DESTINATION ${TAOCPP_SEQUENCES_INSTALL_DOC_DIR})
