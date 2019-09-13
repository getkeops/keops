#pragma once

#include <sstream>
#include <assert.h>

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
  static int VAL(int m) {
    return -1;
  }

  // helpers to print the pack to the standard output
  static void PrintAll(::std::stringstream &str) {}
  static void PrintComma(::std::stringstream &str) {}
  static void PrintId(::std::stringstream &str) {
    str << "pack< >";
  }

  // [].append(M) = [M]
  template<int M>
  using PUTLEFT = pack<M>;

  // Furthermore, the empty package :
  static const int SIZE = 0; // Has zero size (number of vectors) ...
  static const int MAX = -1; // max is set to -1 (we assume packs of non negative integers...)
  static const int SUM = 0;  // ... zero sum  (total memory footprint) ...

  // ... is loaded trivially ...
  template<typename TYPE>
  HOST_DEVICE static void load(int i, TYPE *xi, TYPE **px) {}

  // (even with broadcasting batch dimensions!)
  template<typename TYPE>
  HOST_DEVICE static void load(int i, TYPE *xi, TYPE **px, int *offsets) {}

  // ... counts for nothing in the evaluation of a function ...
  template<typename TYPE, class FUN, typename... Args>
  HOST_DEVICE static void call(FUN fun, TYPE *x, Args... args) {
    fun(args...);
  }

  // ... idem ...
  template<class DIMS, typename TYPE, class FUN, typename... Args>
  HOST_DEVICE static void call2(FUN fun, TYPE *x, Args... args) {
    DIMS::call(fun, args...);
  }

  // ... idem ...
  template<class DIMS1, class DIMS2, typename TYPE, class FUN, typename... Args>
  HOST_DEVICE static void call3(FUN fun, TYPE *x, Args... args) {
    DIMS1::template call2<DIMS2>(fun, args...);
  }

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
  static int VAL(int m) {
    if (m)
      return NEXT::VAL(m - 1);
    else
      return FIRST;
  }

  // helpers to print the pack to the standard output
  static void PrintComma(::std::stringstream &str) {
    str << ",";
  }

  static void PrintAll(::std::stringstream &str) {
    str << FIRST;
    NEXT::PrintComma(str);
    NEXT::PrintAll(str);
  }

  static void PrintId(::std::stringstream &str) {
    str << "pack<";
    PrintAll(str);
    str << ">";
  }

  // Operation to append "M" at the head of the list
  template<int M>
  using PUTLEFT = pack<M, N, NS...>;

  static const int SIZE = 1 + sizeof...(NS);         // The number of vectors in pack<N,NS...>
  typedef pack<NS...> NEXT;                          // "NEXT" is the tail of our list of vectors.
  static const int MAX = static_max(N, NEXT::MAX);   // get the max of values
  static const int
      SUM = N + NEXT::SUM;                           // The total "memory footprint" of pack<N,NS...> is computed recursively.

  // Loads the i-th element of the (global device memory pointer) px to the "array" xi.
  template<typename TYPE>
  HOST_DEVICE static void load(int i, TYPE *xi, TYPE **px) {
    assert(xi != nullptr);
    assert(px != nullptr);
    /*
     * px is an "array" of pointers to data arrays of appropriate sizes.
     * That is, px[0] = *px     is a pointer to a TYPE array of size Ni * FIRST
     * Then,    px[1] = *(px+1) is a pointer to a TYPE array of size Ni * NEXT::FIRST; etc.
     *
     * (where Ni is the max value of "i" you should expect)
     * Obviously, we do not make any sanity check... so beware of illicit memory accesses !
     */
    // Using pythonic syntax, we can describe our loading procedure as follows :
    for (int k = 0; k < FIRST; k++) {
      assert(&((*px)[i * FIRST + k]) != nullptr);
      xi[k] = (*px)[i * FIRST + k];                 // First, load the i-th line of px[0]  -> xi[ 0 : FIRST ].
    }
    NEXT::load(i, xi + FIRST, px + 1);              // Then,  load the i-th line of px[1:] -> xi[ FIRST : ] (recursively)
  }

  // Idem, but with variable-dependent offsets; this is critical for broadcasting
  // batch dimensions in the *_ranges reduction routines:
  template<typename TYPE>
  HOST_DEVICE static void load(int i, TYPE *xi, TYPE **px, int *offsets) {
    assert(xi != nullptr);
    assert(px != nullptr);
    int true_i = offsets[0] + i;
    // Using pythonic syntax, we can describe our loading procedure as follows :
    for (int k = 0; k < FIRST; k++) {
      assert(&((*px)[true_i * FIRST + k]) != nullptr);
      xi[k] = (*px)[true_i * FIRST + k];            // First, load the i-th line of px[0]  -> xi[ 0 : FIRST ].
    }
    NEXT::load(i,
               xi + FIRST,
               px + 1,
               offsets + 1);                        // Then,  load the i-th line of px[1:] -> xi[ FIRST : ] (recursively)
  }

  // call(fun, [x1, x2, x3], arg1, arg2 ) will end up executing fun( arg1, arg2, x1, x2, x3 ).
  template<typename TYPE, class FUN, typename... Args>
  HOST_DEVICE static void call(FUN fun, TYPE *x, Args... args) {
    NEXT::call(fun, x + FIRST, args..., x);         // Append [x[0:FIRST]] to the list of arguments, then iterate.
  }

  // Idem, with a template on DIMS. This allows you to call fun with
  // two "packed" variables (x_i and y_j) as first inputs.
  // call2(fun, [x1, x2], [y1, y2], arg1 ) will end up executing fun(arg1, x1, x2, y1, y2).
  template<class DIMS, typename TYPE, class FUN, typename... Args>
  HOST_DEVICE static void call2(FUN fun, TYPE *x, Args... args) {
    NEXT::template call2<DIMS>(fun, x + FIRST, args..., x);
  }

  // Idem, with a double template on DIMS. This allows you to call fun with
  // three "packed" variables
  template<class DIMS1, class DIMS2, typename TYPE, class FUN, typename... Args>
  HOST_DEVICE static void call3(FUN fun, TYPE *x, Args... args) {
    NEXT::template call3<DIMS1, DIMS2>(fun, x + FIRST, args..., x);
  }

  // Out of a long  list of pointers, extract the ones which "belong" to the current pack
  // and put them into a pointer array px.
  template<typename TYPE, typename... Args>
  static void getlist(TYPE **px, Args... args) {
    *px = Get<FIRST>(args...);
    NEXT::getlist(px + 1, args...);
  }

};



// USEFUL METHODS ===================================================================================

// Templated call
template<class DIMSX, class DIMSY, class DIMSP, typename TYPE, class FUN, typename... Args>
HOST_DEVICE void call(FUN fun, TYPE *x, Args... args) {
  DIMSX::template call3<DIMSY, DIMSP>(fun, x, args...);
}

template<class INDS, typename TYPE, typename... Args>
void getlist(TYPE **px, Args... args) {
  INDS::getlist(px, args...);
}

// Loads the i-th "line" of px to xi.
template<class DIMS, typename TYPE>
HOST_DEVICE void load(int i, TYPE *xi, TYPE **px) {
  DIMS::load(i, xi, px);
}

// Loads the i-th "line" of px to xi, with offsets (used when broadcasting batch dimensions)
template<class DIMS, typename TYPE>
HOST_DEVICE void load(int i, TYPE *xi, TYPE **px, int *offsets) {
  DIMS::load(i, xi, px, offsets);
}

}