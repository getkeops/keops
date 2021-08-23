// Copyright (c) 2015-2019 Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/sequences/

#include <tao/seq/make_integer_sequence.hpp>
#include <tao/seq/sum.hpp>

int main()
{
   using namespace tao::seq;

   static_assert( sum< int >::value == 0, "oops" );

   static_assert( sum< index_sequence<> >::value == 0, "oops" );

   static_assert( sum< int, 0 >::value == 0, "oops" );
   static_assert( sum< int, 0, 0, 0, 0, 0 >::value == 0, "oops" );
   static_assert( sum< int, 0, 1, -1, 0, -1, 0, 1 >::value == 0, "oops" );

   static_assert( sum< unsigned, 3, 1, 5, 0, 1 >::value == 10, "oops" );

   static_assert( sum< index_sequence< 0 > >::value == 0, "oops" );
   static_assert( sum< index_sequence< 42 > >::value == 42, "oops" );

   static_assert( sum< index_sequence< 4, 9, 1, 123, 2 > >::value == 139, "oops" );

   static_assert( sum< make_index_sequence< 10 > >::value == 45, "oops" );
}
