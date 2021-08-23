// Copyright (c) 2015-2019 Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/sequences/

#include <tao/seq/integer_sequence.hpp>
#include <tao/seq/select.hpp>

#include <type_traits>

int main()
{
   using namespace tao::seq;

   using S = integer_sequence< int, 4, 7, -2, 0, 3 >;
   using U = integer_sequence< unsigned, 42 >;

   static_assert( select< 0, S >::value == 4, "oops" );
   static_assert( select< 1, S >::value == 7, "oops" );
   static_assert( select< 2, S >::value == -2, "oops" );
   static_assert( select< 3, S >::value == 0, "oops" );
   static_assert( select< 4, S >::value == 3, "oops" );
   static_assert( std::is_same< select< 0, S >::value_type, int >::value, "oops" );

   static_assert( select< 0, U >::value == 42, "oops" );
   static_assert( std::is_same< select< 0, U >::value_type, unsigned >::value, "oops" );
}
