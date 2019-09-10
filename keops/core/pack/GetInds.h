#pragma once

#include "core/pack/UnivPack.h"

namespace keops {


// Get the list of indices (useful for univpacks of abstract Variables) -------------------------
// GetInds( [X1, X3, Y2] ) = [1, 3, 2]
template < class UPACK >
struct GetInds_Alias {                                  // GetInds( [Xi, ...] )
  using a = typename UPACK::NEXT;
  using c = typename GetInds_Alias<a>::type;
  using type = typename c::template PUTLEFT<UPACK::FIRST::N>; // = [i] + GetInds( [...] )
};

template <>
struct GetInds_Alias< univpack<> > { // GetInds( [] )
  using type = pack<>;
};        // = []

template < class UPACK >
using GetInds = typename GetInds_Alias< UPACK >::type;



}