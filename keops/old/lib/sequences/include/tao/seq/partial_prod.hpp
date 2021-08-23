// Copyright (c) 2019 Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/sequences/

#ifndef TAO_SEQ_PARTIAL_PROD_HPP
#define TAO_SEQ_PARTIAL_PROD_HPP

#include <cstddef>

#include "make_integer_sequence.hpp"
#include "prod.hpp"

namespace tao
{
   namespace seq
   {
      namespace impl
      {
         template< std::size_t, typename S, typename = make_index_sequence< S::size() > >
         struct partial_prod;

         template< std::size_t I, typename T, T... Ns, std::size_t... Is >
         struct partial_prod< I, integer_sequence< T, Ns... >, index_sequence< Is... > >
            : seq::prod< T, ( ( Is < I ) ? Ns : 1 )... >
         {
            static_assert( I <= sizeof...( Is ), "tao::seq::partial_prod<I, S>: I is out of range" );
         };

      }  // namespace impl

      template< std::size_t I, typename T, T... Ns >
      struct partial_prod
         : impl::partial_prod< I, integer_sequence< T, Ns... > >
      {
      };

      template< std::size_t I, typename T, T... Ns >
      struct partial_prod< I, integer_sequence< T, Ns... > >
         : impl::partial_prod< I, integer_sequence< T, Ns... > >
      {
      };

   }  // namespace seq

}  // namespace tao

#endif
