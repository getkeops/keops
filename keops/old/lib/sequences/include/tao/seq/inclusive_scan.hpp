// Copyright (c) 2015-2019 Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/sequences/

#ifndef TAO_SEQ_INCLUSIVE_SCAN_HPP
#define TAO_SEQ_INCLUSIVE_SCAN_HPP

#include "exclusive_scan.hpp"
#include "integer_sequence.hpp"

namespace tao
{
   namespace seq
   {
      template< typename OP, typename T, T... Ns >
      struct inclusive_scan;

      template< typename OP, typename T >
      struct inclusive_scan< OP, T >
      {
         using type = integer_sequence< T >;
      };

      template< typename OP, typename T, T N, T... Ns >
      struct inclusive_scan< OP, T, N, Ns... >
         : exclusive_scan< OP, T, N, Ns..., N >
      {
      };

      template< typename OP, typename T, T... Ns >
      struct inclusive_scan< OP, integer_sequence< T, Ns... > >
         : inclusive_scan< OP, T, Ns... >
      {
      };

      template< typename OP, typename T, T... Ns >
      using inclusive_scan_t = typename inclusive_scan< OP, T, Ns... >::type;

   }  // namespace seq

}  // namespace tao

#endif
