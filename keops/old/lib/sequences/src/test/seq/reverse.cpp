// Copyright (c) 2019 Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/sequences/

#include <type_traits>

#include <tao/seq/integer_sequence.hpp>
#include <tao/seq/reverse.hpp>

int main()
{
   using namespace tao::seq;

   static_assert( std::is_same< reverse_t< long >, integer_sequence< long > >::value, "oops" );
   static_assert( std::is_same< reverse_t< int, 1 >, integer_sequence< int, 1 > >::value, "oops" );
   static_assert( std::is_same< reverse_t< int, 1, 2, 3, 4, 5 >, integer_sequence< int, 5, 4, 3, 2, 1 > >::value, "oops" );
   static_assert( std::is_same< reverse_t< index_sequence< 1, 4, 3, 7, 5 > >, index_sequence< 5, 7, 3, 4, 1 > >::value, "oops" );
}
