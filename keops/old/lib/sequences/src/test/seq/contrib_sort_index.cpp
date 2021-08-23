// Copyright (c) 2019 Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/sequences/

#include <tao/seq/contrib/sort_index.hpp>
#include <tao/seq/functional.hpp>
#include <tao/seq/integer_sequence.hpp>

#include <type_traits>

int main()
{
   using namespace tao::seq;

   using S = integer_sequence< int, 39, 2, -4, 10 >;
   using R = index_sequence< 3, 1, 0, 2 >;

   static_assert( std::is_same< sort_index_t< op::less, S >, R >::value, "oops" );

   static_assert( std::is_same< sort_index_t< op::less, int, 39, 2, -4, 10 >, R >::value, "oops" );
}
