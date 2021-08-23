// Copyright (c) 2015-2019 Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/sequences/

#include <tao/seq/integer_sequence.hpp>
#include <tao/seq/tail.hpp>

#include <type_traits>

int main()
{
   using namespace tao::seq;

   static_assert( std::is_same< tail_t< integer_sequence< int, 0 > >, integer_sequence< int > >::value, "oops" );
   static_assert( std::is_same< tail_t< integer_sequence< unsigned, 0 > >, integer_sequence< unsigned > >::value, "oops" );

   static_assert( std::is_same< tail_t< integer_sequence< int, 0, 1 > >, integer_sequence< int, 1 > >::value, "oops" );
   static_assert( std::is_same< tail_t< integer_sequence< unsigned, 0, 1 > >, integer_sequence< unsigned, 1 > >::value, "oops" );

   static_assert( std::is_same< tail_t< integer_sequence< int, 0, 1, 2, 3, 4, 5 > >, integer_sequence< int, 1, 2, 3, 4, 5 > >::value, "oops" );
   static_assert( std::is_same< tail_t< integer_sequence< unsigned, 0, 1, 2, 3, 4, 5 > >, integer_sequence< unsigned, 1, 2, 3, 4, 5 > >::value, "oops" );
}
