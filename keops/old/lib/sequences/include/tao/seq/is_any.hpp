// Copyright (c) 2015-2019 Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/sequences/

#ifndef TAO_SEQ_IS_ANY_HPP
#define TAO_SEQ_IS_ANY_HPP

#include "config.hpp"

#ifndef TAO_SEQ_FOLD_EXPRESSIONS
#include "is_all.hpp"
#endif

#include <type_traits>

namespace tao
{
   namespace seq
   {

#ifdef TAO_SEQ_FOLD_EXPRESSIONS

      template< bool... Bs >
      struct is_any : std::integral_constant< bool, ( Bs || ... ) >
      {
      };

#else

      template< bool... Bs >
      struct is_any : std::integral_constant< bool, !is_all< !Bs... >::value >
      {
      };

#endif

   }  // namespace seq

}  // namespace tao

#endif
