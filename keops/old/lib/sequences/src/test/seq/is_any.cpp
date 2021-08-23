// Copyright (c) 2015-2019 Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/sequences/

#include <tao/seq/is_any.hpp>

int main()
{
   using namespace tao::seq;

   static_assert( !is_any<>::value, "oops" );

   static_assert( is_any< true >::value, "oops" );
   static_assert( !is_any< false >::value, "oops" );

   static_assert( is_any< true, true >::value, "oops" );
   static_assert( is_any< true, false >::value, "oops" );
   static_assert( is_any< false, true >::value, "oops" );
   static_assert( !is_any< false, false >::value, "oops" );

   static_assert( is_any< true, true, true, true, true, true >::value, "oops" );
   static_assert( !is_any< false, false, false, false, false, false >::value, "oops" );

   static_assert( is_any< false, true, true, true, true, true >::value, "oops" );
   static_assert( is_any< true, false, true, true, true, true >::value, "oops" );
   static_assert( is_any< true, true, false, true, true, true >::value, "oops" );
   static_assert( is_any< true, true, true, false, true, true >::value, "oops" );
   static_assert( is_any< true, true, true, true, false, true >::value, "oops" );
   static_assert( is_any< true, true, true, true, true, false >::value, "oops" );

   static_assert( is_any< true, false, false, false, false, false >::value, "oops" );
   static_assert( is_any< false, true, false, false, false, false >::value, "oops" );
   static_assert( is_any< false, false, true, false, false, false >::value, "oops" );
   static_assert( is_any< false, false, false, true, false, false >::value, "oops" );
   static_assert( is_any< false, false, false, false, true, false >::value, "oops" );
   static_assert( is_any< false, false, false, false, false, true >::value, "oops" );
}
