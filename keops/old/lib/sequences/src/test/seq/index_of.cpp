// Copyright (c) 2017-2019 Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/sequences/

#include <tao/seq/index_of.hpp>
#include <tao/seq/make_integer_sequence.hpp>

int main()
{
   using namespace tao::seq;

   static_assert( index_of< int, 42, 42 >::value == 0, "oops" );

   static_assert( index_of< int, 42, 42, 0, 1, 2, 3, 4 >::value == 0, "oops" );
   static_assert( index_of< int, 42, 0, 1, 42, 3, 4 >::value == 2, "oops" );
   static_assert( index_of< int, 42, 0, 1, 2, 3, 4, 42 >::value == 5, "oops" );

   static_assert( index_of< make_integer_sequence< int, 43 >, 42 >::value == 42, "oops" );
}
