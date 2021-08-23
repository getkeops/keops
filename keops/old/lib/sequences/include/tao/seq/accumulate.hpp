// Copyright (c) 2015-2019 Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/sequences/

#ifndef TAO_SEQ_ACCUMULATE_HPP
#define TAO_SEQ_ACCUMULATE_HPP

#include <type_traits>

#include "config.hpp"
#include "integer_sequence.hpp"

namespace tao
{
   namespace seq
   {

#ifdef TAO_SEQ_FOLD_EXPRESSIONS

      namespace impl
      {
         template< typename OP, typename T, T N >
         struct wrap_accumulate
         {
         };

         template< typename OP, typename T, T R, T L >
         constexpr auto operator+( std::integral_constant< T, R >, wrap_accumulate< OP, T, L > ) noexcept
         {
            return typename OP::template apply< T, R, L >();
         }

         template< typename OP, typename T, T N, T... Ns >
         constexpr auto accumulate() noexcept
         {
            return ( std::integral_constant< T, N >() + ... + wrap_accumulate< OP, T, Ns >() );
         }

      }  // namespace impl

      template< typename OP, typename T, T... Ns >
      struct accumulate
         : decltype( impl::accumulate< OP, T, Ns... >() )
      {
      };

#else

      namespace impl
      {
         template< bool >
         struct accumulate
         {
            template< typename, typename T, T N >
            using apply = std::integral_constant< T, N >;
         };

         template<>
         struct accumulate< false >
         {
            template< typename OP, typename T, T N0, T N1, T... Ns >
            using apply = typename accumulate< sizeof...( Ns ) == 0 >::template apply< OP, T, OP::template apply< T, N0, N1 >::value, Ns... >;
         };

      }  // namespace impl

      template< typename OP, typename T, T... Ns >
      struct accumulate
         : impl::accumulate< sizeof...( Ns ) == 1 >::template apply< OP, T, Ns... >
      {
      };

#endif

      template< typename OP, typename T, T... Ns >
      struct accumulate< OP, integer_sequence< T, Ns... > >
         : accumulate< OP, T, Ns... >
      {
      };

   }  // namespace seq

}  // namespace tao

#endif
