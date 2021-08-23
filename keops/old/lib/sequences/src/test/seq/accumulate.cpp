// Copyright (c) 2015-2019 Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/sequences/

#include <tao/seq/accumulate.hpp>
#include <tao/seq/functional.hpp>

int main()
{
   using namespace tao::seq;

   static_assert( accumulate< op::plus, int, 1 >::value == 1, "oops" );
   static_assert( accumulate< op::plus, int, 1, 2 >::value == 3, "oops" );
   static_assert( accumulate< op::plus, int, 1, 2, 3 >::value == 6, "oops" );
   static_assert( accumulate< op::plus, int, 1, 2, 3, 4 >::value == 10, "oops" );

   static_assert( accumulate< op::multiplies, int, 1, 2, 3 >::value == 6, "oops" );
   static_assert( accumulate< op::multiplies, int, 1, 2, 3, 4 >::value == 24, "oops" );

   static_assert( accumulate< op::plus, int, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 >::value == 55, "oops" );
   static_assert( accumulate< op::plus, int, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 >::value == 66, "oops" );
}
