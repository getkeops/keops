// Copyright (c) 2017-2019 Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/sequences/

#include <tao/seq/contains.hpp>
#include <tao/seq/make_integer_sequence.hpp>

int main()
{
   using namespace tao::seq;

   static_assert( !contains< int, 42 >::value, "oops" );

   static_assert( !contains< int, 42, 0 >::value, "oops" );
   static_assert( contains< int, 42, 42 >::value, "oops" );

   static_assert( !contains< int, 42, 0, 1, 2, 3, 4 >::value, "oops" );
   static_assert( contains< int, 42, 42, 0, 1, 2, 3, 4 >::value, "oops" );
   static_assert( contains< int, 42, 0, 1, 42, 3, 4 >::value, "oops" );
   static_assert( contains< int, 42, 0, 1, 2, 3, 4, 42 >::value, "oops" );

   static_assert( !contains< make_integer_sequence< int, 0 >, 42 >::value, "oops" );
   static_assert( !contains< make_integer_sequence< int, 42 >, 42 >::value, "oops" );
   static_assert( contains< make_integer_sequence< int, 43 >, 42 >::value, "oops" );
}
