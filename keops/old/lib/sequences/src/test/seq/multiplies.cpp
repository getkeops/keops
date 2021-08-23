// Copyright (c) 2019 Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/sequences/

#include <tao/seq/integer_sequence.hpp>
#include <tao/seq/multiplies.hpp>

#include <type_traits>

int main()
{
   using namespace tao::seq;

   using A = index_sequence< 0, 5, 7, 1 >;
   using B = index_sequence< 4, 5, 6, 7 >;

   using R = index_sequence< 0, 25, 42, 7 >;

   static_assert( std::is_same< multiplies_t< A, B >, R >::value, "oops" );
}
