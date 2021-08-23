// Copyright (c) 2015-2019 Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/sequences/

#include <tao/seq/is_all.hpp>

int main()
{
   using namespace tao::seq;

   static_assert( is_all<>::value, "oops" );

   static_assert( is_all< true >::value, "oops" );
   static_assert( !is_all< false >::value, "oops" );

   static_assert( is_all< true, true >::value, "oops" );
   static_assert( !is_all< true, false >::value, "oops" );
   static_assert( !is_all< false, true >::value, "oops" );
   static_assert( !is_all< false, false >::value, "oops" );

   static_assert( is_all< true, true, true, true, true, true >::value, "oops" );

   static_assert( !is_all< false, true, true, true, true, true >::value, "oops" );
   static_assert( !is_all< true, false, true, true, true, true >::value, "oops" );
   static_assert( !is_all< true, true, false, true, true, true >::value, "oops" );
   static_assert( !is_all< true, true, true, false, true, true >::value, "oops" );
   static_assert( !is_all< true, true, true, true, false, true >::value, "oops" );
   static_assert( !is_all< true, true, true, true, true, false >::value, "oops" );

   static_assert( !is_all< true, false, false, false, false, false >::value, "oops" );
   static_assert( !is_all< false, true, false, false, false, false >::value, "oops" );
   static_assert( !is_all< false, false, true, false, false, false >::value, "oops" );
   static_assert( !is_all< false, false, false, true, false, false >::value, "oops" );
   static_assert( !is_all< false, false, false, false, true, false >::value, "oops" );
   static_assert( !is_all< false, false, false, false, false, true >::value, "oops" );
}
