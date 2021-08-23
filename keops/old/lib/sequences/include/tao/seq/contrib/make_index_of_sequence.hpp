// Copyright (c) 2019 Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/sequences/

#ifndef TAO_SEQ_CONTRIB_MAKE_INDEX_OF_SEQUENCE_HPP
#define TAO_SEQ_CONTRIB_MAKE_INDEX_OF_SEQUENCE_HPP

#include <type_traits>

#include "../index_of.hpp"
#include "../integer_sequence.hpp"

namespace tao
{
   namespace seq
   {
      template< typename, typename >
      struct make_index_of_sequence;

      template< typename TA, TA... As, typename TB, TB... Bs >
      struct make_index_of_sequence< integer_sequence< TA, As... >, integer_sequence< TB, Bs... > >
      {
         using type = index_sequence< index_of< typename std::common_type< TA, TB >::type, Bs, As... >::value... >;
      };

      template< typename A, typename B >
      using make_index_of_sequence_t = typename make_index_of_sequence< A, B >::type;

   }  // namespace seq

}  // namespace tao

#endif
