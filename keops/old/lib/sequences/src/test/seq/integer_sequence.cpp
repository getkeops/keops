// Copyright (c) 2015-2019 Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/sequences/

#include <tao/seq/integer_sequence.hpp>

#include <type_traits>

int main()
{
   using namespace tao::seq;

   using S0 = integer_sequence< int >;
   using S1 = integer_sequence< int, 0 >;
   using S2 = integer_sequence< int, 5, 7 >;
   using S4 = integer_sequence< int, 0, 1, -1, 42 >;
   using U4 = integer_sequence< unsigned, 0, 1, 7, 42 >;

   static_assert( S0::size() == 0, "oops" );
   static_assert( S1::size() == 1, "oops" );
   static_assert( S2::size() == 2, "oops" );
   static_assert( S4::size() == 4, "oops" );
   static_assert( U4::size() == 4, "oops" );

   static_assert( std::is_same< S0::value_type, int >::value, "oops" );
   static_assert( std::is_same< S4::value_type, int >::value, "oops" );
   static_assert( std::is_same< U4::value_type, unsigned >::value, "oops" );
}
