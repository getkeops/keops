#pragma once

#include <sstream>
#include <assert.h>
#include <climits>

#include "core/pre_headers.h"
#include "core/pack/Get.h"
#include "core/pack/CondType.h"

namespace keops {

// At compilation time, detect the maximum between two values (typically, dimensions)
template < typename T >
static constexpr T static_max(T a, T b) {
  return a < b ? b : a;
}
/////////////////////////////////////////////////
//              PACKS OF VECTORS               //
/////////////////////////////////////////////////


// Define recursively how a "package" of variables should behave.
// Packages are handled as list of vectors, stored in a contiguous memory space.
// They are meant to represent (location,), (location, normal), (signal, location)
// or any other kind of "pointwise" feature that is looped over by convolution operations.
//
// A package is instantiated using a
//        "typedef pack<1,DIM,DIM> DIMSX;"
// syntax (for example), which means that DIMSX is the dimension
// of a "3-uple" of (TYPE) variables, one scalar then two vectors
// of size DIM.

// The EMPTY package : ==============================================================================
template<int... NS>
struct pack {

  // DIMSX::VAL(2) is the size of its 3rd vector (starts at 0) (in the example above, DIM).
  // Therefore, EMPTY::VAL(n) should never be called : we return -1 as an error signal.
  HOST_DEVICE static int VAL(int m) {
    return -1;
  }

  // helpers to print the pack to the standard output
  static void PrintAll(::std::ostream &str) {}
  static void PrintComma(::std::ostream &str) {}
  static void PrintId(::std::ostream &str) {
    str << "pack< >";
  }

  // [].append(M) = [M]
  template<int M>
  using PUTLEFT = pack<M>;

  // Furthermore, the empty package :
  static const int SIZE = 0; // Has zero size (number of vectors) ...
  static const int MIN = INT_MAX;
  static const int MAX = -1; // max is set to -1 (we assume packs of non negative integers...)
  static const int SUM = 0;  // ... zero sum  (total memory footprint) ...

  // ... does not have anything to give to a list of variables.
  template<typename TYPE, typename... Args>
  static void getlist(TYPE **px, Args... args) {}

};

// A non-EMPTY package, recursively defined as : ====================================================
// "The concatenation of a vector of size N, and a (possibly empty) package."
template<int N, int... NS>
struct pack<N, NS...> {
  static const int FIRST = N;    // Size of its first element.

  // DIMSX::VAL(2) = size of its 3rd vector (we start counting at 0).
  HOST_DEVICE static int VAL(int m) {
    if (m)
      return NEXT::VAL(m - 1);
    else
      return FIRST;
  }

  // helpers to print the pack to the standard output
  static void PrintComma(::std::ostream &str) {
    str << ",";
  }

  static void PrintAll(::std::ostream &str) {
    str << FIRST;
    NEXT::PrintComma(str);
    NEXT::PrintAll(str);
  }

  static void PrintId(::std::ostream &str) {
    str << "pack<";
    PrintAll(str);
    str << ">";
  }

  // Operation to append "M" at the head of the list
  template<int M>
  using PUTLEFT = pack<M, N, NS...>;

  static const int SIZE = 1 + sizeof...(NS);         // The number of vectors in pack<N,NS...>
  typedef pack<NS...> NEXT;                          // "NEXT" is the tail of our list of vectors.
  static const int MIN = -static_max(-N, -NEXT::MIN);   // get the min of values
  static const int MAX = static_max(N, NEXT::MAX);   // get the max of values
  static const int
      SUM = N + NEXT::SUM;                           // The total "memory footprint" of pack<N,NS...> is computed recursively.

  // Out of a long  list of pointers, extract the ones which "belong" to the current pack
  // and put them into a pointer array px.
  template<typename TYPE, typename... Args>
  static void getlist(TYPE **px, Args... args) {
    *px = Get<FIRST>(args...);
    NEXT::getlist(px + 1, args...);
  }

};



// USEFUL METHODS ===================================================================================



template<class INDS, typename TYPE, typename... Args>
void getlist(TYPE **px, Args... args) {
  INDS::getlist(px, args...);
}


// unpacking for variadic templates (NB. Args is supposed to be a sequence of TYPE*)
template <typename TYPE, typename... Args>
typename std::enable_if<sizeof...(Args) == 0>::type unpack(TYPE **p, Args... args) { }

template < typename TYPE, typename FirstArg, typename... Args >
void unpack(TYPE **p, FirstArg firstarg, Args... args)
{
    p[0] = firstarg;
    unpack(p+1, args...);
}

}
