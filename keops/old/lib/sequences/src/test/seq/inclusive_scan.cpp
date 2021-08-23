// Copyright (c) 2015-2019 Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/sequences/

#include <tao/seq/functional.hpp>
#include <tao/seq/inclusive_scan.hpp>
#include <tao/seq/integer_sequence.hpp>

#include <type_traits>

int main()
{
   using namespace tao::seq;

   static_assert( std::is_same< inclusive_scan_t< op::plus, unsigned >, integer_sequence< unsigned > >::value, "oops" );
   static_assert( std::is_same< inclusive_scan_t< op::plus, int, 2, 3, -1, 0, 4, -1, 1, 1 >, integer_sequence< int, 2, 5, 4, 4, 8, 7, 8, 9 > >::value, "oops" );

   using S = integer_sequence< int, 2, 3, -1, 0, 4, -1, 1, 1 >;
   static_assert( std::is_same< inclusive_scan_t< op::plus, S >, integer_sequence< int, 2, 5, 4, 4, 8, 7, 8, 9 > >::value, "oops" );

   static_assert( std::is_same< inclusive_scan_t< op::multiplies, int, 3, 1, 4, 1, 5, 9, 2, 6 >, integer_sequence< int, 3, 3, 12, 12, 60, 540, 1080, 6480 > >::value, "oops" );
}
