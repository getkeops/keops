// Copyright (c) 2015-2019 Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/sequences/

#ifndef TAO_SEQ_INTEGER_SEQUENCE_HPP
#define TAO_SEQ_INTEGER_SEQUENCE_HPP

#include <cstddef>
#include <utility>

#include "config.hpp"

namespace tao
{
   namespace seq
   {

#ifdef TAO_SEQ_USE_STD_INTEGER_SEQUENCE

      using std::index_sequence;
      using std::integer_sequence;  // NOLINT(misc-unused-using-decls)

#else

      template< typename T, T... Ns >
      struct integer_sequence
      {
         using value_type = T;

         static constexpr std::size_t size() noexcept
         {
            return sizeof...( Ns );
         }
      };

      template< std::size_t... Ns >
      using index_sequence = integer_sequence< std::size_t, Ns... >;

#endif

   }  // namespace seq

}  // namespace tao

#endif
