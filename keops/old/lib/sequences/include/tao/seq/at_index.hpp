// Copyright (c) 2015-2019 Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/sequences/

#ifndef TAO_SEQ_AT_INDEX_HPP
#define TAO_SEQ_AT_INDEX_HPP

#include <cstddef>

#include "config.hpp"
#include "make_integer_sequence.hpp"

namespace tao
{
   namespace seq
   {

#ifdef TAO_SEQ_TYPE_PACK_ELEMENT

      template< std::size_t I, typename... Ts >
      using at_index_t = __type_pack_element< I, Ts... >;

      template< std::size_t I, typename... Ts >
      struct at_index
      {
         using type = at_index_t< I, Ts... >;
      };

#else

      // based on http://talesofcpp.fusionfenix.com/post-22/true-story-efficient-packing

      namespace impl
      {
         template< std::size_t, typename T >
         struct indexed
         {
            using type = T;
         };

         template< typename, typename... Ts >
         struct indexer;

         template< std::size_t... Is, typename... Ts >
         struct indexer< index_sequence< Is... >, Ts... >
            : indexed< Is, Ts >...
         {
         };

#if( __cplusplus >= 201402L )
         template< typename... Ts >
         constexpr impl::indexer< index_sequence_for< Ts... >, Ts... > index_value{};
#endif

         template< std::size_t I, typename T >
         indexed< I, T > select( const indexed< I, T >& );

      }  // namespace impl

#if( __cplusplus >= 201402L )

      template< std::size_t I, typename... Ts >
      using at_index = decltype( impl::select< I >( impl::index_value< Ts... > ) );

#else

      template< std::size_t I, typename... Ts >
      using at_index = decltype( impl::select< I >( impl::indexer< index_sequence_for< Ts... >, Ts... >() ) );

#endif

#ifndef _MSC_VER

      template< std::size_t I, typename... Ts >
      using at_index_t = typename at_index< I, Ts... >::type;

#else

      namespace impl
      {
         template< typename T >
         struct get_type
         {
            using type = typename T::type;
         };

      }  // namespace impl

      template< std::size_t I, typename... Ts >
      using at_index_t = typename impl::get_type< at_index< I, Ts... > >::type;

#endif

#endif

   }  // namespace seq

}  // namespace tao

#endif
