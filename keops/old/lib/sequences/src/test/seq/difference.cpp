// Copyright (c) 2019 Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/sequences/

#include <tao/seq/difference.hpp>
#include <tao/seq/integer_sequence.hpp>

#include <type_traits>

int main()
{
   using namespace tao::seq;

   using A = integer_sequence< int, 1, 2, 3, 4, 5, 6 >;
   using B = integer_sequence< int, 2, 5 >;
   using R = integer_sequence< int, 1, 3, 4, 6 >;

   static_assert( std::is_same< difference_t< A, B >, R >::value, "oops" );
}
