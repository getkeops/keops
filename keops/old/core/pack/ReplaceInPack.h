#pragma once

namespace keops {

///////////////////////////////////////////
// Replace a value at position N in pack //
///////////////////////////////////////////

template< class P, int V, int N >
struct ReplaceInPack_Alias {
  using NEXTPACK = typename P::NEXT;
  using type = typename ReplaceInPack_Alias< NEXTPACK, V, N - 1 >::type::template PUTLEFT< P::FIRST >;
};

template< class P, int V >
struct ReplaceInPack_Alias< P, V, 0 > {
  using type = typename P::NEXT::template PUTLEFT< V >;
};

template< class P, int V, int N >
using ReplaceInPack = typename ReplaceInPack_Alias< P, V, N >::type;

}