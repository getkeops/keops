#pragma once

#include "core/pack/UnivPack.h"

namespace keops {


// Search position in a pack -------------------------------------------------------------------------
// IndVal( [ 1, 4, 2, 0,...], 2 ) = 3
template < class INTPACK, int N >    // IndVal( [C, ...], N)     ( C != N )
struct IndVal_Alias {                 // = 1 + IndVal( [...], N)
  static const int ind = 1 + IndVal_Alias< typename INTPACK::NEXT, N >::ind;
};

template < int N, int... NS >
struct IndVal_Alias< pack<N,NS...>, N > { // IndVal( [N, ...], N)
static const int ind = 0;
};        // = 0

template < int N >
struct IndVal_Alias< pack<>, N > {       // IndVal( [], N )
  static const int ind = 0;
};        // = 0

template < class INTPACK, int N >
struct IndVal {				// Use as IndVal<Intpack, N>::value
  static const int value = IndVal_Alias<INTPACK,N>::ind;
};

}