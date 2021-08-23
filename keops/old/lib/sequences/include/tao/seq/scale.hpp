// Copyright (c) 2015-2019 Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/sequences/

#ifndef TAO_SEQ_SCALE_HPP
#define TAO_SEQ_SCALE_HPP

#include "integer_sequence.hpp"
#include "sequence_helper.hpp"

namespace tao
{
   namespace seq
   {
      namespace impl
      {
         template< typename T >
         struct scale
         {
            template< T S, T... Ns >
            struct impl
            {
               using type = integer_sequence< T, S * Ns... >;
            };
         };

         template< typename T, T... Ns >
         struct scale< integer_sequence< T, Ns... > >
         {
            template< T S >
            struct impl
            {
               using type = integer_sequence< T, S * Ns... >;
            };
         };

      }  // namespace impl

      template< typename T, typename impl::element_type< T >::type S, T... Ns >
      using scale = typename impl::scale< T >::template impl< S, Ns... >;

      template< typename T, typename impl::element_type< T >::type S, T... Ns >
      using scale_t = typename scale< T, S, Ns... >::type;

   }  // namespace seq

}  // namespace tao

#endif
