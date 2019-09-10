#pragma once

namespace keops {

// Access the n-th element of an univpack -------------------------------------------------------
// Val( [ x0, x1, x2, ...], i ) = xi
template < class UPACK, int N >                                     // Val([C, ...], N)  (N > 0)
struct Val_Alias {
  using a = typename UPACK::NEXT;
  using type = typename Val_Alias<a, N-1 >::type; // = Val([...], N-1)
};

template < class UPACK >
struct Val_Alias< UPACK, 0 > {                    // Val([C, ...], 0)
  using type = typename UPACK::FIRST;
};                                                 // = C

template < class UPACK, int N >
using Val = typename Val_Alias< UPACK, N >::type;

}