// Copyright (c) 2015-2019 Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/sequences/

#ifndef TAO_SEQ_MINUS_HPP
#define TAO_SEQ_MINUS_HPP

#include "functional.hpp"
#include "zip.hpp"

namespace tao
{
   namespace seq
   {
      template< typename A, typename B >
      using minus = zip< op::minus, A, B >;

      template< typename A, typename B >
      using minus_t = typename minus< A, B >::type;

   }  // namespace seq

}  // namespace tao

#endif
