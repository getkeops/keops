// Copyright (c) 2015-2019 Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/sequences/

#ifndef TAO_SEQ_PARTIAL_SUM_HPP
#define TAO_SEQ_PARTIAL_SUM_HPP

#include <cstddef>

#include "make_integer_sequence.hpp"
#include "sum.hpp"

namespace tao
{
   namespace seq
   {
      namespace impl
      {
         template< std::size_t, typename S, typename = make_index_sequence< S::size() > >
         struct partial_sum;

         template< std::size_t I, typename T, T... Ns, std::size_t... Is >
         struct partial_sum< I, integer_sequence< T, Ns... >, index_sequence< Is... > >
            : seq::sum< T, ( ( Is < I ) ? Ns : 0 )... >
         {
            static_assert( I <= sizeof...( Is ), "tao::seq::partial_sum<I, S>: I is out of range" );
         };

      }  // namespace impl

      template< std::size_t I, typename T, T... Ns >
      struct partial_sum
         : impl::partial_sum< I, integer_sequence< T, Ns... > >
      {
      };

      template< std::size_t I, typename T, T... Ns >
      struct partial_sum< I, integer_sequence< T, Ns... > >
         : impl::partial_sum< I, integer_sequence< T, Ns... > >
      {
      };

   }  // namespace seq

}  // namespace tao

#endif
