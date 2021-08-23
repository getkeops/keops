// Copyright (c) 2015-2019 Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/sequences/

#ifndef TAO_SEQ_SELECT_HPP
#define TAO_SEQ_SELECT_HPP

#include <cstddef>
#include <utility>

#include "at_index.hpp"
#include "integer_sequence.hpp"

namespace tao
{
   namespace seq
   {
      template< std::size_t I, typename T, T... Ns >
      struct select
         : at_index_t< I, std::integral_constant< T, Ns >... >
      {
      };

      template< std::size_t I, typename T, T... Ns >
      struct select< I, integer_sequence< T, Ns... > >
         : select< I, T, Ns... >
      {
      };

   }  // namespace seq

}  // namespace tao

#endif
