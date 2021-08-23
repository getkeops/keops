// Copyright (c) 2015-2019 Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/sequences/

#include <tao/seq/make_integer_sequence.hpp>
#include <tao/seq/max.hpp>

int main()
{
   using namespace tao::seq;

   static_assert( max< int, 1 >::value == 1, "oops" );
   static_assert( max< int, 1, 0 >::value == 1, "oops" );
   static_assert( max< int, 0, 1 >::value == 1, "oops" );
   static_assert( max< int, 1, -1 >::value == 1, "oops" );
   static_assert( max< int, -1, 1 >::value == 1, "oops" );
   static_assert( max< int, -1, 1, 2 >::value == 2, "oops" );
   static_assert( max< int, -1, 2, 1 >::value == 2, "oops" );
   static_assert( max< int, 2, -1, 1 >::value == 2, "oops" );
   static_assert( max< int, 0, 1, 2, -1, 1, 0, -1, 1 >::value == 2, "oops" );
   static_assert( max< make_index_sequence< 10 > >::value == 9, "oops" );
}
