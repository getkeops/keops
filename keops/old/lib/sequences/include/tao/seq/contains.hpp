// Copyright (c) 2017-2019 Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/sequences/

#ifndef TAO_SEQ_CONTAINS_HPP
#define TAO_SEQ_CONTAINS_HPP

#include "integer_sequence.hpp"
#include "is_any.hpp"
#include "sequence_helper.hpp"

namespace tao
{
   namespace seq
   {
      namespace impl
      {
         template< typename T >
         struct contains
         {
            template< T N, T... Ns >
            using type = is_any< ( N == Ns )... >;
         };

         template< typename T, T... Ns >
         struct contains< integer_sequence< T, Ns... > >
         {
            template< T N >
            using type = is_any< ( N == Ns )... >;
         };

      }  // namespace impl

      template< typename T, typename impl::element_type< T >::type N, T... Ns >
      using contains = typename impl::contains< T >::template type< N, Ns... >;

   }  // namespace seq

}  // namespace tao

#endif
