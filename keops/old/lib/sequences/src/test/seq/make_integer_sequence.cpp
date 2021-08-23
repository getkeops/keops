// Copyright (c) 2015-2019 Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/sequences/

#include <tao/seq/make_integer_sequence.hpp>

#include <type_traits>

int main()
{
   using namespace tao::seq;

   static_assert( std::is_same< make_integer_sequence< int, 0 >, integer_sequence< int > >::value, "oops" );
   static_assert( std::is_same< make_integer_sequence< unsigned, 0 >, integer_sequence< unsigned > >::value, "oops" );

   static_assert( std::is_same< make_integer_sequence< int, 1 >, integer_sequence< int, 0 > >::value, "oops" );
   static_assert( std::is_same< make_integer_sequence< unsigned, 1 >, integer_sequence< unsigned, 0 > >::value, "oops" );

   static_assert( std::is_same< make_integer_sequence< int, 2 >, integer_sequence< int, 0, 1 > >::value, "oops" );
   static_assert( std::is_same< make_integer_sequence< unsigned, 2 >, integer_sequence< unsigned, 0, 1 > >::value, "oops" );

   static_assert( std::is_same< make_integer_sequence< int, 5 >, integer_sequence< int, 0, 1, 2, 3, 4 > >::value, "oops" );
   static_assert( std::is_same< make_integer_sequence< unsigned, 5 >, integer_sequence< unsigned, 0, 1, 2, 3, 4 > >::value, "oops" );

   static_assert( std::is_same< make_integer_sequence< int, 10 >, integer_sequence< int, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 > >::value, "oops" );
   static_assert( std::is_same< make_integer_sequence< unsigned, 10 >, integer_sequence< unsigned, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 > >::value, "oops" );

   static_assert( std::is_same< make_index_sequence< 0 >, index_sequence<> >::value, "oops" );
   static_assert( std::is_same< make_index_sequence< 1 >, index_sequence< 0 > >::value, "oops" );
   static_assert( std::is_same< make_index_sequence< 10 >, index_sequence< 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 > >::value, "oops" );

   static_assert( make_index_sequence< 0 >::size() == 0, "oops" );
   static_assert( make_index_sequence< 1 >::size() == 1, "oops" );
   static_assert( make_index_sequence< 10 >::size() == 10, "oops" );
   static_assert( make_index_sequence< 100 >::size() == 100, "oops" );

#ifndef _MSC_VER  // Visual C++ complains about the symbol length
   static_assert( make_index_sequence< 1000 >::size() == 1000, "oops" );
   static_assert( make_index_sequence< 10000 >::size() == 10000, "oops" );
   static_assert( make_index_sequence< 100000 >::size() == 100000, "oops" );
#endif
}
