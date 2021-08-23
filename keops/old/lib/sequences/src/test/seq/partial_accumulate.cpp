// Copyright (c) 2019 Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/sequences/

#include <tao/seq/functional.hpp>
#include <tao/seq/partial_accumulate.hpp>

int main()
{
   using namespace tao::seq;

   static_assert( partial_accumulate< op::plus, 1, int, 1 >::value == 1, "oops" );
   static_assert( partial_accumulate< op::plus, 2, int, 1, 2, 3 >::value == 3, "oops" );

   static_assert( partial_accumulate< op::plus, 10, int, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 >::value == 55, "oops" );
   static_assert( partial_accumulate< op::plus, 10, int, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 >::value == 55, "oops" );
   static_assert( partial_accumulate< op::plus, 11, int, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 >::value == 66, "oops" );
}
