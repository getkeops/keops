// Copyright (c) 2015-2019 Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/sequences/

#ifndef TAO_SEQ_MAKE_INTEGER_SEQUENCE_HPP
#define TAO_SEQ_MAKE_INTEGER_SEQUENCE_HPP

#include <cstddef>
#include <utility>

#include "config.hpp"
#include "integer_sequence.hpp"

namespace tao
{
   namespace seq
   {

#ifdef TAO_SEQ_USE_STD_MAKE_INTEGER_SEQUENCE

      using std::index_sequence_for;
      using std::make_index_sequence;
      using std::make_integer_sequence;

#else

      namespace impl
      {
         // we have four instantiations of generate_sequence<>, independent of T or N.
         // V is the current bit, E is the end marker - if true, this is the last step.
         template< bool V, bool E >
         struct generate_sequence;

         // last step: generate final integer sequence
         template<>
         struct generate_sequence< false, true >
         {
            template< typename T, T M, T N, std::size_t S, T... Ns >
            using f = integer_sequence< T, Ns... >;
         };

         template<>
         struct generate_sequence< true, true >
         {
            template< typename T, T M, T N, std::size_t S, T... Ns >
            using f = integer_sequence< T, Ns..., S >;
         };

         // intermediate step: double existing values, append one more if V is set.
         template<>
         struct generate_sequence< false, false >
         {
            template< typename T, T M, T N, std::size_t S, T... Ns >
            using f = typename generate_sequence< ( N & ( M / 2 ) ) != 0, ( M / 2 ) == 0 >::template f< T, M / 2, N, 2 * S, Ns..., ( Ns + S )... >;
         };

         template<>
         struct generate_sequence< true, false >
         {
            template< typename T, T M, T N, std::size_t S, T... Ns >
            using f = typename generate_sequence< ( N & ( M / 2 ) ) != 0, ( M / 2 ) == 0 >::template f< T, M / 2, N, 2 * S + 1, Ns..., ( Ns + S )..., 2 * S >;
         };

         // the final sequence per T/N should be memoized, it will probably be used multiple times.
         // also checks the limit and starts the above generator properly.
         template< typename T, T N >
         struct memoize_sequence
         {
            static_assert( N < T( 1 << 20 ), "N too large" );
            using type = typename generate_sequence< false, false >::template f< T, ( N < T( 1 << 1 ) ) ? T( 1 << 1 ) : ( N < T( 1 << 2 ) ) ? T( 1 << 2 ) : ( N < T( 1 << 3 ) ) ? T( 1 << 3 ) : ( N < T( 1 << 4 ) ) ? T( 1 << 4 ) : ( N < T( 1 << 5 ) ) ? T( 1 << 5 ) : ( N < T( 1 << 6 ) ) ? T( 1 << 6 ) : ( N < T( 1 << 7 ) ) ? T( 1 << 7 ) : ( N < T( 1 << 8 ) ) ? T( 1 << 8 ) : ( N < T( 1 << 9 ) ) ? T( 1 << 9 ) : ( N < T( 1 << 10 ) ) ? T( 1 << 10 ) : T( 1 << 20 ), N, 0 >;
         };

      }  // namespace impl

      template< typename T, T N >
      using make_integer_sequence = typename impl::memoize_sequence< T, N >::type;

      template< std::size_t N >
      using make_index_sequence = make_integer_sequence< std::size_t, N >;

      template< typename... Ts >
      using index_sequence_for = make_index_sequence< sizeof...( Ts ) >;

#endif

   }  // namespace seq

}  // namespace tao

#endif
