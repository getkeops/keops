#pragma once

#include "core/pack/UnivPack.h"

namespace keops {

// Get the list of dimensions. ------------------------------------------------------------------
// GetDims([a, b, c]) = [dim_a, dim_b, dim_c]                (works for univpacks)
template < class UPACK >
struct GetDims_Alias {
  using a = typename UPACK::NEXT;
  using c = typename GetDims_Alias< a >::type;
  using type = typename c::template PUTLEFT< UPACK::FIRST::DIM >;
};

template <>
struct GetDims_Alias< univpack<> > {
  using type = pack<>;
};

template < class UPACK >
using GetDims = typename GetDims_Alias< UPACK >::type;



}