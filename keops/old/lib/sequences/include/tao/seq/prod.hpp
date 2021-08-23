// Copyright (c) 2019 Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/sequences/

#ifndef TAO_SEQ_PROD_HPP
#define TAO_SEQ_PROD_HPP

#include "config.hpp"
#include "integer_sequence.hpp"

#ifdef TAO_SEQ_FOLD_EXPRESSIONS

#include <type_traits>

#else

#include "functional.hpp"
#include "reduce.hpp"

#endif

namespace tao
{
   namespace seq
   {

#ifdef TAO_SEQ_FOLD_EXPRESSIONS

      template< typename T, T... Ns >
      struct prod
         : std::integral_constant< T, ( T( 1 ) * ... * Ns ) >
      {
      };

#else

      template< typename T, T... Ns >
      struct prod
         : reduce< op::multiplies, T, T( 1 ), Ns... >
      {
      };

#endif

      template< typename T, T... Ns >
      struct prod< integer_sequence< T, Ns... > >
         : prod< T, Ns... >
      {
      };

   }  // namespace seq

}  // namespace tao

#endif
