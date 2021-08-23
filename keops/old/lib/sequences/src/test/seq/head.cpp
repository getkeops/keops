// Copyright (c) 2015-2019 Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/sequences/

#include <tao/seq/head.hpp>
#include <tao/seq/integer_sequence.hpp>

#include <type_traits>

int main()
{
   using namespace tao::seq;

   static_assert( head< int, 0 >::value == 0, "oops" );
   static_assert( head< int, 0, 42 >::value == 0, "oops" );
   static_assert( head< int, 0, 1, 2, 3, 4, 5 >::value == 0, "oops" );

   static_assert( head< int, 42 >::value == 42, "oops" );
   static_assert( head< int, 42, 0 >::value == 42, "oops" );
   static_assert( head< int, 42, 1, 2, 3, 4, 5 >::value == 42, "oops" );

   static_assert( head< index_sequence< 0 > >::value == 0, "oops" );
   static_assert( head< index_sequence< 0, 42 > >::value == 0, "oops" );
   static_assert( head< index_sequence< 0, 1, 2, 3, 4, 5 > >::value == 0, "oops" );

   static_assert( std::is_same< head< integer_sequence< int, 0 > >::value_type, int >::value, "oops" );
   static_assert( std::is_same< head< integer_sequence< unsigned, 0 > >::value_type, unsigned >::value, "oops" );
}
