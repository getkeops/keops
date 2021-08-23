// Copyright (c) 2019 Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/sequences/

#ifndef TAO_SEQ_DIFFERENCE_HPP
#define TAO_SEQ_DIFFERENCE_HPP

#include <type_traits>

#include "concatenate.hpp"
#include "contains.hpp"
#include "integer_sequence.hpp"
#include "sequence_helper.hpp"

namespace tao
{
   namespace seq
   {
      template< typename, typename >
      struct difference;

      template< typename TA, TA... As, typename TB, TB... Bs >
      struct difference< integer_sequence< TA, As... >, integer_sequence< TB, Bs... > >
      {
         using CT = typename std::common_type< TA, TB >::type;

         template< CT N >
         using check = contains< CT, N, Bs... >;

         using type = concatenate_t< impl::conditional_t< check< As >::value, integer_sequence< CT >, integer_sequence< CT, As > >... >;
      };

      template< typename A, typename B >
      using difference_t = typename difference< A, B >::type;

   }  // namespace seq

}  // namespace tao

#endif
