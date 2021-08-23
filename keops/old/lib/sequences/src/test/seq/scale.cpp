// Copyright (c) 2015-2019 Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/sequences/

#include <tao/seq/make_integer_sequence.hpp>
#include <tao/seq/scale.hpp>

#include <type_traits>

int main()
{
   using namespace tao::seq;

   static_assert( std::is_same< scale_t< int, 0 >, integer_sequence< int > >::value, "oops" );
   static_assert( std::is_same< scale_t< unsigned, 0 >, integer_sequence< unsigned > >::value, "oops" );

   static_assert( std::is_same< scale_t< int, 1, 0 >, integer_sequence< int, 0 > >::value, "oops" );
   static_assert( std::is_same< scale_t< unsigned, 1, 0 >, integer_sequence< unsigned, 0 > >::value, "oops" );

   static_assert( std::is_same< scale_t< int, 2, 0, 1, 2, 3 >, integer_sequence< int, 0, 2, 4, 6 > >::value, "oops" );
   static_assert( std::is_same< scale_t< int, 0, 0, 1, 2, 3 >, integer_sequence< int, 0, 0, 0, 0 > >::value, "oops" );
   static_assert( std::is_same< scale_t< int, -2, 0, 1, 2, 3 >, integer_sequence< int, 0, -2, -4, -6 > >::value, "oops" );
   static_assert( std::is_same< scale_t< int, -2, 0, -1, 2, -3 >, integer_sequence< int, 0, 2, -4, 6 > >::value, "oops" );

   using S = make_integer_sequence< int, 4 >;

   static_assert( std::is_same< scale_t< S, 1 >, S >::value, "oops" );

   static_assert( std::is_same< scale_t< S, 2 >, integer_sequence< int, 0, 2, 4, 6 > >::value, "oops" );
   static_assert( std::is_same< scale_t< S, 0 >, integer_sequence< int, 0, 0, 0, 0 > >::value, "oops" );
   static_assert( std::is_same< scale_t< S, -2 >, integer_sequence< int, 0, -2, -4, -6 > >::value, "oops" );
}
