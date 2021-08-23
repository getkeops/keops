// Copyright (c) 2015-2019 Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/sequences/

#ifndef TAO_SEQ_MAP_HPP
#define TAO_SEQ_MAP_HPP

#include <cstddef>

#include "integer_sequence.hpp"
#include "select.hpp"

namespace tao
{
   namespace seq
   {
      template< typename, typename >
      struct map;

      template< std::size_t... Ns, typename M >
      struct map< index_sequence< Ns... >, M >
      {
         using type = integer_sequence< typename M::value_type, select< Ns, M >::value... >;
      };

      template< typename S, typename M >
      using map_t = typename map< S, M >::type;

   }  // namespace seq

}  // namespace tao

#endif
