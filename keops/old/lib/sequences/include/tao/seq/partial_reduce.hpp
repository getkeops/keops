// Copyright (c) 2019 Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/sequences/

#ifndef TAO_SEQ_PARTIAL_REDUCE_HPP
#define TAO_SEQ_PARTIAL_REDUCE_HPP

#include <cstddef>

#include "first.hpp"
#include "reduce.hpp"

namespace tao
{
   namespace seq
   {
      template< typename OP, std::size_t I, typename T, T... Ns >
      using partial_reduce = reduce< OP, first_t< I, T, Ns... > >;

   }  // namespace seq

}  // namespace tao

#endif
