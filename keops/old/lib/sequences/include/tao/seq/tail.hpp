// Copyright (c) 2015-2019 Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/sequences/

#ifndef TAO_SEQ_TAIL_HPP
#define TAO_SEQ_TAIL_HPP

#include "integer_sequence.hpp"

namespace tao
{
   namespace seq
   {
      template< typename T, T... Ns >
      struct tail;

      template< typename T, T N, T... Ns >
      struct tail< T, N, Ns... >
      {
         using type = integer_sequence< T, Ns... >;
      };

      template< typename T, T... Ns >
      struct tail< integer_sequence< T, Ns... > >
         : tail< T, Ns... >
      {
      };

      template< typename T, T... Ns >
      using tail_t = typename tail< T, Ns... >::type;

   }  // namespace seq

}  // namespace tao

#endif
