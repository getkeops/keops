// Copyright (c) 2015-2019 Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/sequences/

#include <tao/seq/at_index.hpp>

#include <type_traits>

int main()
{
   using namespace tao::seq;

   static_assert( std::is_same< at_index_t< 0, void >, void >::value, "oops" );
   static_assert( std::is_same< at_index_t< 0, void* >, void* >::value, "oops" );

   static_assert( std::is_same< at_index_t< 0, int >, int >::value, "oops" );
   static_assert( std::is_same< at_index_t< 0, int* >, int* >::value, "oops" );
   static_assert( std::is_same< at_index_t< 0, int& >, int& >::value, "oops" );

   static_assert( std::is_same< at_index_t< 0, const int >, const int >::value, "oops" );
   static_assert( std::is_same< at_index_t< 0, const int* >, const int* >::value, "oops" );
   static_assert( std::is_same< at_index_t< 0, const int& >, const int& >::value, "oops" );

   static_assert( std::is_same< at_index_t< 0, int( long, double ) >, int( long, double ) >::value, "oops" );

   static_assert( !std::is_same< at_index_t< 0, void, int >, int >::value, "oops" );
   static_assert( std::is_same< at_index_t< 1, void, int >, int >::value, "oops" );

   // clang-format off
   static_assert( !std::is_same< at_index_t< 53,
                  void, void, void, void, void, void, void, void,
                  void, void, void, void, void, void, void, void,
                  void, void, void, void, void, void, void, void,
                  void, void, void, void, void, void, void, void,
                  void, void, void, void, void, void, void, void,
                  void, void, void, void, void, void, void, void,
                  void, void, void, void, long, void, void, void >, long >::value, "oops" );
   // clang-format on
}
