// Copyright (c) 2015-2019 Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/sequences/

#include <tao/seq/integer_sequence.hpp>
#include <tao/seq/map.hpp>

#include <type_traits>

int main()
{
   using namespace tao::seq;

   using S = index_sequence< 2, 0, 1, 1, 4, 3, 1, 2, 0 >;
   using M = integer_sequence< int, 11, 88, 42, 0, 99 >;

   static_assert( std::is_same< map_t< index_sequence<>, M >, integer_sequence< int > >::value, "oops" );
   static_assert( std::is_same< map_t< S, M >, integer_sequence< int, 42, 11, 88, 88, 99, 0, 88, 42, 11 > >::value, "oops" );
   static_assert( std::is_same< map_t< S, M >, integer_sequence< int, 42, 11, 88, 88, 99, 0, 88, 42, 11 > >::value, "oops" );
}
