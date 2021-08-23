// Copyright (c) 2015-2019 Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/sequences/

#ifndef TAO_SEQ_PLUS_HPP
#define TAO_SEQ_PLUS_HPP

#include "functional.hpp"
#include "zip.hpp"

namespace tao
{
   namespace seq
   {
      template< typename A, typename B >
      using plus = zip< op::plus, A, B >;

      template< typename A, typename B >
      using plus_t = typename plus< A, B >::type;

   }  // namespace seq

}  // namespace tao

#endif
