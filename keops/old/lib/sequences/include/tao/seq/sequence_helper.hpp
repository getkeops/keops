// Copyright (c) 2017-2019 Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/sequences/

#ifndef TAO_SEQ_SEQUENCE_HELPER_HPP
#define TAO_SEQ_SEQUENCE_HELPER_HPP

#include <cstddef>
#include <utility>

#include "integer_sequence.hpp"

namespace tao
{
   namespace seq
   {
      namespace impl
      {
         template< typename T >
         struct element_type
         {
            using type = T;
         };

         template< typename T, T... Ns >
         struct element_type< integer_sequence< T, Ns... > >
         {
            using type = T;
         };

         template< typename T, T... Ns >
         struct sequence_size
            : std::integral_constant< std::size_t, sizeof...( Ns ) >
         {
         };

         template< typename T, T... Ns >
         struct sequence_size< integer_sequence< T, Ns... > >
            : std::integral_constant< std::size_t, sizeof...( Ns ) >
         {
         };

         template< bool >
         struct conditional
         {
            template< typename T, typename >
            using type = T;
         };

         template<>
         struct conditional< false >
         {
            template< typename, typename F >
            using type = F;
         };

         template< bool C, typename T, typename F >
         using conditional_t = typename conditional< C >::template type< T, F >;

      }  // namespace impl

   }  // namespace seq

}  // namespace tao

#endif
