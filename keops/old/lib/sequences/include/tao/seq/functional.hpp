// Copyright (c) 2019 Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/sequences/

#ifndef TAO_SEQ_FUNCTIONAL_HPP
#define TAO_SEQ_FUNCTIONAL_HPP

#include <type_traits>

namespace tao
{
   namespace seq
   {
      namespace op
      {
         struct plus
         {
            template< typename T, T A, T B >
            using apply = std::integral_constant< T, A + B >;
         };

         struct minus
         {
            template< typename T, T A, T B >
            using apply = std::integral_constant< T, A - B >;
         };

         struct multiplies
         {
            template< typename T, T A, T B >
            using apply = std::integral_constant< T, A * B >;
         };

         struct divides
         {
            template< typename T, T A, T B >
            using apply = std::integral_constant< T, A / B >;
         };

         struct modulus
         {
            template< typename T, T A, T B >
            using apply = std::integral_constant< T, A % B >;
         };

         struct equal_to
         {
            template< typename T, T A, T B >
            using apply = std::integral_constant< bool, A == B >;
         };

         struct not_equal_to
         {
            template< typename T, T A, T B >
            using apply = std::integral_constant< bool, A != B >;
         };

         struct greater
         {
            template< typename T, T A, T B >
            using apply = std::integral_constant< bool, ( A > B ) >;
         };

         struct less
         {
            template< typename T, T A, T B >
            using apply = std::integral_constant< bool, ( A < B ) >;
         };

         struct greater_equal
         {
            template< typename T, T A, T B >
            using apply = std::integral_constant< bool, ( A >= B ) >;
         };

         struct less_equal
         {
            template< typename T, T A, T B >
            using apply = std::integral_constant< bool, ( A <= B ) >;
         };

         struct logical_and
         {
            template< typename T, T A, T B >
            using apply = std::integral_constant< bool, A && B >;
         };

         struct logical_or
         {
            template< typename T, T A, T B >
            using apply = std::integral_constant< bool, A || B >;
         };

         struct bit_and
         {
            template< typename T, T A, T B >
            using apply = std::integral_constant< T, A & B >;
         };

         struct bit_or
         {
            template< typename T, T A, T B >
            using apply = std::integral_constant< T, A | B >;
         };

         struct bit_xor
         {
            template< typename T, T A, T B >
            using apply = std::integral_constant< T, A ^ B >;
         };

         struct min
         {
            template< typename T, T A, T B >
            using apply = std::integral_constant< T, ( A < B ) ? A : B >;
         };

         struct max
         {
            template< typename T, T A, T B >
            using apply = std::integral_constant< T, ( A > B ) ? A : B >;
         };

      }  // namespace op

   }  // namespace seq

}  // namespace tao

#endif
