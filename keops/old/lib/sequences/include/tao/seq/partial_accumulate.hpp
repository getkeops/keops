// Copyright (c) 2019 Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/sequences/

#ifndef TAO_SEQ_PARTIAL_ACCUMULATE_HPP
#define TAO_SEQ_PARTIAL_ACCUMULATE_HPP

#include <cstddef>

#include "accumulate.hpp"
#include "first.hpp"
#include "integer_sequence.hpp"

namespace tao
{
   namespace seq
   {
      template< typename OP, std::size_t I, typename T, T... Ns >
      using partial_accumulate = accumulate< OP, first_t< I, T, Ns... > >;

   }  // namespace seq

}  // namespace tao

#endif
