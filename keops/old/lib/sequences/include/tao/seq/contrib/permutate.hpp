// Copyright (c) 2019 Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/sequences/

#ifndef TAO_SEQ_CONTRIB_PERMUTATE_HPP
#define TAO_SEQ_CONTRIB_PERMUTATE_HPP

#include "../make_integer_sequence.hpp"
#include "../map.hpp"
#include "make_index_of_sequence.hpp"

namespace tao
{
   namespace seq
   {
      template< typename I >
      using inverse_t = make_index_of_sequence_t< I, make_index_sequence< I::size() > >;

      template< typename I, typename S >
      using permutate = map< inverse_t< I >, S >;

      template< typename I, typename S >
      using permutate_t = typename permutate< I, S >::type;

   }  // namespace seq

}  // namespace tao

#endif
