// Copyright (c) 2015-2019 Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/sequences/

#include <tao/seq/integer_sequence.hpp>
#include <tao/seq/minus.hpp>

#include <type_traits>

int main()
{
   using namespace tao::seq;

   using S0 = integer_sequence< int >;
   using LL0 = integer_sequence< long long >;

   static_assert( minus_t< S0, S0 >::size() == 0, "oops" );
   static_assert( minus_t< S0, LL0 >::size() == 0, "oops" );
   static_assert( minus_t< LL0, S0 >::size() == 0, "oops" );
   static_assert( minus_t< LL0, LL0 >::size() == 0, "oops" );

   static_assert( std::is_same< minus_t< S0, S0 >, S0 >::value, "oops" );
   static_assert( std::is_same< minus_t< S0, LL0 >, LL0 >::value, "oops" );
   static_assert( std::is_same< minus_t< LL0, S0 >, LL0 >::value, "oops" );
   static_assert( std::is_same< minus_t< LL0, LL0 >, LL0 >::value, "oops" );

   using S3 = integer_sequence< int, 1, 2, -3 >;
   using LL3 = integer_sequence< long long, 4, 7, 8 >;

   static_assert( std::is_same< minus_t< S3, LL3 >, integer_sequence< long long, -3, -5, -11 > >::value, "oops" );
   static_assert( std::is_same< minus_t< LL3, S3 >, integer_sequence< long long, 3, 5, 11 > >::value, "oops" );
   static_assert( std::is_same< minus_t< LL3, LL3 >, integer_sequence< long long, 0, 0, 0 > >::value, "oops" );
}
