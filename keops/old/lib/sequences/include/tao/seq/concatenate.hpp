// Copyright (c) 2015-2019 Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/sequences/

#ifndef TAO_SEQ_CONCATENATE_HPP
#define TAO_SEQ_CONCATENATE_HPP

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
         template< typename T >
         struct wrap_concat
         {
         };

         template< typename TA, TA... As, typename TB, TB... Bs >
         constexpr auto operator+( integer_sequence< TA, As... >, wrap_concat< integer_sequence< TB, Bs... > > ) noexcept
         {
            return integer_sequence< typename std::common_type< TA, TB >::type, As..., Bs... >();
         }

         template< typename T, typename... Ts >
         constexpr auto concat() noexcept
         {
            return ( T() + ... + wrap_concat< Ts >() );
         }

      }  // namespace impl

      template< typename... Ts >
      using concatenate_t = decltype( impl::concat< Ts... >() );

      template< typename... Ts >
      struct concatenate
      {
         using type = concatenate_t< Ts... >;
      };

#else

      template< typename... >
      struct concatenate;

      template< typename T, T... Ns >
      struct concatenate< integer_sequence< T, Ns... > >
      {
         using type = integer_sequence< T, Ns... >;
      };

      // TODO: Improve recursion
      template< typename TA, TA... As, typename TB, TB... Bs, typename... Ts >
      struct concatenate< integer_sequence< TA, As... >, integer_sequence< TB, Bs... >, Ts... >
         : concatenate< integer_sequence< typename std::common_type< TA, TB >::type, As..., Bs... >, Ts... >
      {
      };

      template< typename... Ts >
      using concatenate_t = typename concatenate< Ts... >::type;

#endif

   }  // namespace seq

}  // namespace tao

#endif
