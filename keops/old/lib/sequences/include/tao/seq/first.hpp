// Copyright (c) 2019 Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/sequences/

#ifndef TAO_SEQ_FIRST_HPP
#define TAO_SEQ_FIRST_HPP

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
         struct first;

         template< std::size_t... Is, typename T, T... Ns >
         struct first< index_sequence< Is... >, T, Ns... >
         {
            template< std::size_t I >
            using element = seq::select< I, T, Ns... >;

            using type = integer_sequence< T, element< Is >::value... >;
         };

      }  // namespace impl

      template< std::size_t I, typename T, T... Ns >
      struct first
         : impl::first< make_index_sequence< I >, T, Ns... >
      {
      };

      template< std::size_t I, typename T, T... Ns >
      struct first< I, integer_sequence< T, Ns... > >
         : first< I, T, Ns... >
      {
      };

      template< std::size_t I, typename T, T... Ns >
      using first_t = typename first< I, T, Ns... >::type;

   }  // namespace seq

}  // namespace tao

#endif
