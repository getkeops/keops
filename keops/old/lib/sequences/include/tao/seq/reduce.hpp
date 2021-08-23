// Copyright (c) 2019 Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/sequences/

#ifndef TAO_SEQ_REDUCE_HPP
#define TAO_SEQ_REDUCE_HPP

#include "config.hpp"

#ifdef TAO_SEQ_FOLD_EXPRESSIONS

#include "accumulate.hpp"

#else

#include <cstddef>
#include <type_traits>

#include "make_integer_sequence.hpp"
#include "select.hpp"

#endif

namespace tao
{
   namespace seq
   {

#ifdef TAO_SEQ_FOLD_EXPRESSIONS

      template< typename OP, typename T, T... Ns >
      using reduce = accumulate< OP, T, Ns... >;

#else

      namespace impl
      {
         template< typename OP, bool, typename, typename T, T... >
         struct reducer;

         template< typename OP, std::size_t... Is, typename T, T... Ns >
         struct reducer< OP, false, index_sequence< Is... >, T, Ns... >
         {
            template< std::size_t I >
            using subsel = seq::select< I, T, Ns... >;

            using type = integer_sequence< T, OP::template apply< T, subsel< 2 * Is >::value, subsel< 2 * Is + 1 >::value >::value... >;
         };

         template< typename OP, std::size_t... Is, typename T, T N, T... Ns >
         struct reducer< OP, true, index_sequence< Is... >, T, N, Ns... >
         {
            template< std::size_t I >
            using subsel = seq::select< I, T, Ns... >;

            using type = integer_sequence< T, N, OP::template apply< T, subsel< 2 * Is >::value, subsel< 2 * Is + 1 >::value >::value... >;
         };

      }  // namespace impl

      template< typename, typename T, T... >
      struct reduce;

      template< typename OP, typename T, T N >
      struct reduce< OP, T, N >
         : std::integral_constant< T, N >
      {
      };

      template< typename OP, typename T, T... Ns >
      struct reduce
         : reduce< OP, typename impl::reducer< OP, sizeof...( Ns ) % 2 == 1, make_index_sequence< sizeof...( Ns ) / 2 >, T, Ns... >::type >
      {
      };

      template< typename OP, typename T, T... Ns >
      struct reduce< OP, integer_sequence< T, Ns... > >
         : reduce< OP, T, Ns... >
      {
      };

#endif

   }  // namespace seq

}  // namespace tao

#endif
