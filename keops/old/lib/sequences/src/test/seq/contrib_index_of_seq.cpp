// Copyright (c) 2019 Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/sequences/

#include <tao/seq/contrib/make_index_of_sequence.hpp>
#include <tao/seq/integer_sequence.hpp>

#include <type_traits>

int main()
{
   using namespace tao::seq;

   using S = integer_sequence< int, 7, -2, 3, 0, 4 >;
   using T = integer_sequence< int, 4, 7, -2, 0, 3 >;

   using R = index_sequence< 4, 0, 1, 3, 2 >;

   static_assert( std::is_same< make_index_of_sequence_t< S, T >, R >::value, "oops" );
}
