// Copyright (c) 2015-2019 Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/sequences/

#ifndef TAO_SEQ_MAKE_INTEGER_RANGE_HPP
#define TAO_SEQ_MAKE_INTEGER_RANGE_HPP

#include <cstddef>

#include "make_integer_sequence.hpp"

namespace tao
{
   namespace seq
   {
      namespace impl
      {
         template< typename T, T Begin, T Steps, bool Increase, T Delta = T( 1 ), typename = make_integer_sequence< T, Steps > >
         struct generate_range;

         template< typename T, T B, T S, T D, T... Ns >
         struct generate_range< T, B, S, true, D, integer_sequence< T, Ns... > >
         {
            using type = integer_sequence< T, B + D * Ns... >;
         };

         template< typename T, T B, T S, T D, T... Ns >
         struct generate_range< T, B, S, false, D, integer_sequence< T, Ns... > >
         {
            using type = integer_sequence< T, B - D * Ns... >;
         };

      }  // namespace impl

      template< typename T, T N, T M >
      using make_integer_range = typename impl::generate_range< T, N, ( N <= M ) ? ( M - N ) : ( N - M ), ( N <= M ) >::type;

      template< std::size_t N, std::size_t M >
      using make_index_range = make_integer_range< std::size_t, N, M >;

   }  // namespace seq

}  // namespace tao

#endif
