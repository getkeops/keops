// Copyright (c) 2019 Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/sequences/

#include <tao/seq/functional.hpp>
#include <tao/seq/make_integer_sequence.hpp>
#include <tao/seq/sort.hpp>

#include <type_traits>

int main()
{
   using namespace tao::seq;

   static_assert( std::is_same< sort_t< op::less, index_sequence<> >, index_sequence<> >::value, "oops" );
   static_assert( std::is_same< sort_t< op::less, index_sequence< 42 > >, index_sequence< 42 > >::value, "oops" );

   static_assert( std::is_same< sort_t< op::less, int, 7, -2, 3, 0, 4 >, integer_sequence< int, -2, 0, 3, 4, 7 > >::value, "oops" );

   // duplicate entries are allowed!
   using I = index_sequence< 39, 10, 2, 4, 10, 2 >;
   using R = index_sequence< 2, 2, 4, 10, 10, 39 >;
   static_assert( std::is_same< sort_t< op::less, I >, R >::value, "oops" );

   using L = index_sequence< 54, 12, 45, 20, 91, 99, 16, 73, 62, 19, 30, 34, 67, 49, 18, 11, 25, 22, 86, 87, 44, 2, 21, 79, 15, 41, 55, 84, 48, 75, 23, 68, 88, 94, 40, 72, 71, 69, 57, 35, 90, 4, 56, 6, 10, 80, 64, 50, 28, 1, 52, 27, 5, 0, 78, 32, 7, 24, 14, 13, 65, 70, 83, 81, 51, 33, 37, 66, 47, 29, 42, 93, 89, 9, 46, 77, 58, 60, 3, 76, 92, 63, 39, 96, 53, 26, 31, 17, 38, 85, 98, 59, 43, 97, 95, 36, 74, 61, 8, 82 >;
   static_assert( std::is_same< sort_t< op::less, L >, make_index_sequence< 100 > >::value, "oops" );
}
