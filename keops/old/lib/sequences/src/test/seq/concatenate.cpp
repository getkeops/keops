// Copyright (c) 2015-2019 Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/sequences/

#include <tao/seq/concatenate.hpp>
#include <tao/seq/integer_sequence.hpp>

#include <type_traits>

int main()
{
   using namespace tao::seq;

   using E = integer_sequence< int >;
   using O = integer_sequence< int, 0 >;
   using A = integer_sequence< int, 1, 2, 3 >;
   using B = integer_sequence< int, 4, 5, 6 >;

   static_assert( std::is_same< concatenate_t< E, E >, E >::value, "oops" );
   static_assert( std::is_same< concatenate_t< E, O >, O >::value, "oops" );
   static_assert( std::is_same< concatenate_t< O, E >, O >::value, "oops" );
   static_assert( std::is_same< concatenate_t< E, A >, A >::value, "oops" );
   static_assert( std::is_same< concatenate_t< A, E >, A >::value, "oops" );
   static_assert( std::is_same< concatenate_t< E, B >, B >::value, "oops" );
   static_assert( std::is_same< concatenate_t< B, E >, B >::value, "oops" );

   static_assert( std::is_same< concatenate_t< O, A >, integer_sequence< int, 0, 1, 2, 3 > >::value, "oops" );
   static_assert( std::is_same< concatenate_t< A, O >, integer_sequence< int, 1, 2, 3, 0 > >::value, "oops" );

   static_assert( std::is_same< concatenate_t< A, B >, integer_sequence< int, 1, 2, 3, 4, 5, 6 > >::value, "oops" );
   static_assert( std::is_same< concatenate_t< B, A >, integer_sequence< int, 4, 5, 6, 1, 2, 3 > >::value, "oops" );

   static_assert( std::is_same< concatenate_t< E, E, E >, E >::value, "oops" );
   static_assert( std::is_same< concatenate_t< E, E, E, E, E, E >, E >::value, "oops" );

   static_assert( std::is_same< concatenate_t< E, A, B, E, A, O >, integer_sequence< int, 1, 2, 3, 4, 5, 6, 1, 2, 3, 0 > >::value, "oops" );
}
