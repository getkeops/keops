// Copyright (c) 2019 Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/sequences/

#include <tao/seq/integer_sequence.hpp>
#include <tao/seq/prod.hpp>

int main()
{
   using namespace tao::seq;

   static_assert( prod< int >::value == 1, "oops" );
   static_assert( prod< int, -1, 4, 5 >::value == -20, "oops" );
   static_assert( prod< index_sequence<> >::value == 1, "oops" );
   static_assert( prod< integer_sequence< int, -1, 4, 5 > >::value == -20, "oops" );
}
