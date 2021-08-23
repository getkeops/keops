// Copyright (c) 2019 Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/sequences/

#ifndef TAO_SEQ_SORT_HPP
#define TAO_SEQ_SORT_HPP

#include <cstddef>
#include <type_traits>

#include "at_index.hpp"
#include "make_integer_sequence.hpp"
#include "sequence_helper.hpp"

namespace tao
{
   namespace seq
   {
      namespace impl
      {
         template< typename... >
         struct merge;

         template< typename OP, typename T, T A, T... As, T... Rs >
         struct merge< OP, integer_sequence< T, A, As... >, integer_sequence< T >, integer_sequence< T, Rs... > >
         {
            using type = integer_sequence< T, Rs..., A, As... >;
         };

         template< typename OP, typename T, T A, T... As, T B, T... Bs, T... Rs >
         struct merge< OP, integer_sequence< T, A, As... >, integer_sequence< T, B, Bs... >, integer_sequence< T, Rs... > >
            : conditional_t< OP::template apply< T, A, B >::value,
                             merge< OP, integer_sequence< T, B, Bs... >, integer_sequence< T, As... >, integer_sequence< T, Rs..., A > >,
                             merge< OP, integer_sequence< T, A, As... >, integer_sequence< T, Bs... >, integer_sequence< T, Rs..., B > > >
         {
         };

         template< bool, typename... >
         struct multi_merge;

         template< typename OP, typename T, typename S >
         struct multi_merge< false, index_sequence<>, OP, T, S >
         {
            using type = S;
         };

         template< std::size_t... Is, typename OP, typename T, typename... Ss >
         struct multi_merge< true, index_sequence< Is... >, OP, T, Ss... >
            : multi_merge< ( ( sizeof...( Is ) % 2 ) == 0 ),
                           make_index_sequence< sizeof...( Is ) / 2 >,
                           OP,
                           T,
                           typename merge< OP, at_index_t< Is * 2, Ss... >, at_index_t< Is * 2 + 1, Ss... >, integer_sequence< T > >::type... >
         {
         };

         template< std::size_t... Is, typename OP, typename T, typename S, typename... Ss >
         struct multi_merge< false, index_sequence< Is... >, OP, T, S, Ss... >
            : multi_merge< ( ( ( sizeof...( Is ) + 1 ) % 2 ) == 0 ),
                           make_index_sequence< ( sizeof...( Is ) + 1 ) / 2 >,
                           OP,
                           T,
                           S,
                           typename merge< OP, at_index_t< Is * 2, Ss... >, at_index_t< Is * 2 + 1, Ss... >, integer_sequence< T > >::type... >
         {
         };

      }  // namespace impl

      template< typename OP, typename T, T... Ns >
      struct sort
         : impl::multi_merge< ( ( sizeof...( Ns ) % 2 ) == 0 ),
                              make_index_sequence< sizeof...( Ns ) / 2 >,
                              OP,
                              T,
                              integer_sequence< T, Ns >... >
      {
      };

      template< typename OP, typename T >
      struct sort< OP, T >
      {
         using type = integer_sequence< T >;
      };

      template< typename OP, typename T, T... Ns >
      struct sort< OP, integer_sequence< T, Ns... > >
         : sort< OP, T, Ns... >
      {
      };

      template< typename OP, typename T, typename impl::element_type< T >::type... Ns >
      using sort_t = typename sort< OP, T, Ns... >::type;

   }  // namespace seq

}  // namespace tao

#endif
