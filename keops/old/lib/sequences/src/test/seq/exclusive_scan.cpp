// Copyright (c) 2015-2019 Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/sequences/

#include <tao/seq/exclusive_scan.hpp>
#include <tao/seq/functional.hpp>
#include <tao/seq/integer_sequence.hpp>

#include <type_traits>

int main()
{
   using namespace tao::seq;

   static_assert( std::is_same< exclusive_scan_t< op::plus, unsigned, 0 >, integer_sequence< unsigned > >::value, "oops" );
   static_assert( std::is_same< exclusive_scan_t< op::plus, int, 0, 2, 3, -1, 0, 4, -1, 1, 1 >, integer_sequence< int, 0, 2, 5, 4, 4, 8, 7, 8 > >::value, "oops" );

   using S = integer_sequence< int, 2, 3, -1, 0, 4, -1, 1, 1 >;
   static_assert( std::is_same< exclusive_scan_t< op::plus, S, 0 >, integer_sequence< int, 0, 2, 5, 4, 4, 8, 7, 8 > >::value, "oops" );
}
