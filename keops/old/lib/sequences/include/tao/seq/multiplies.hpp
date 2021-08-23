// Copyright (c) 2019 Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/sequences/

#ifndef TAO_SEQ_MULTIPLIES_HPP
#define TAO_SEQ_MULTIPLIES_HPP

#include "functional.hpp"
#include "zip.hpp"

namespace tao
{
   namespace seq
   {
      template< typename A, typename B >
      using multiplies = zip< op::multiplies, A, B >;

      template< typename A, typename B >
      using multiplies_t = typename multiplies< A, B >::type;

   }  // namespace seq

}  // namespace tao

#endif
