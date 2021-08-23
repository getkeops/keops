// Copyright (c) 2015-2019 Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/sequences/

#include <tao/seq/integer_sequence.hpp>
#include <tao/seq/make_integer_range.hpp>

#include <type_traits>

int main()
{
   using namespace tao::seq;

   static_assert( std::is_same< make_integer_range< int, 0, 0 >, integer_sequence< int > >::value, "oops" );
   static_assert( std::is_same< make_integer_range< unsigned, 0, 0 >, integer_sequence< unsigned > >::value, "oops" );

   static_assert( std::is_same< make_integer_range< int, 1, 1 >, integer_sequence< int > >::value, "oops" );
   static_assert( std::is_same< make_integer_range< unsigned, 5, 5 >, integer_sequence< unsigned > >::value, "oops" );

   static_assert( std::is_same< make_integer_range< int, 1, 5 >, integer_sequence< int, 1, 2, 3, 4 > >::value, "oops" );
   static_assert( std::is_same< make_integer_range< unsigned, 5, 8 >, integer_sequence< unsigned, 5, 6, 7 > >::value, "oops" );

   static_assert( std::is_same< make_integer_range< int, -3, 3 >, integer_sequence< int, -3, -2, -1, 0, 1, 2 > >::value, "oops" );
   static_assert( std::is_same< make_integer_range< int, 3, -3 >, integer_sequence< int, 3, 2, 1, 0, -1, -2 > >::value, "oops" );

   static_assert( std::is_same< make_index_range< 42, 42 >, index_sequence<> >::value, "oops" );
   static_assert( std::is_same< make_index_range< 3, 7 >, index_sequence< 3, 4, 5, 6 > >::value, "oops" );

   static_assert( std::is_same< make_index_range< 7, 3 >, index_sequence< 7, 6, 5, 4 > >::value, "oops" );
}
