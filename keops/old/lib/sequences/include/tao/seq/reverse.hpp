// Copyright (c) 2019 Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/sequences/

#ifndef TAO_SEQ_REVERSE_HPP
#define TAO_SEQ_REVERSE_HPP

#include <cstddef>

#include "make_integer_sequence.hpp"
#include "select.hpp"

namespace tao
{
   namespace seq
   {
      namespace impl
      {
         template< typename I, typename T, T... Ns >
         struct reverse;

         template< std::size_t... Is, typename T, T... Ns >
         struct reverse< index_sequence< Is... >, T, Ns... >
         {
            template< std::size_t I >
            using element = seq::select< ( sizeof...( Is ) - 1 ) - I, T, Ns... >;

            using type = integer_sequence< T, element< Is >::value... >;
         };

      }  // namespace impl

      template< typename T, T... Ns >
      struct reverse
         : impl::reverse< make_index_sequence< sizeof...( Ns ) >, T, Ns... >
      {
      };

      template< typename T, T... Ns >
      struct reverse< integer_sequence< T, Ns... > >
         : reverse< T, Ns... >
      {
      };

      template< typename T, T... Ns >
      using reverse_t = typename reverse< T, Ns... >::type;

   }  // namespace seq

}  // namespace tao

#endif
