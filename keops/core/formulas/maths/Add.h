#pragma once

#include <sstream>
#include <assert.h>

#include "core/pack/CondType.h"
#include "core/autodiff/BinaryOp.h"
#include "core/formulas/constants/Zero.h"
#include "core/formulas/constants/IntConst.h"
#include "core/formulas/maths/Sum.h"
#include "core/formulas/maths/Scal.h"

#include "core/pre_headers.h"

namespace keops {

// We need some pre-declarations due to the co-dependency of Add and Scal
template < class FA, class FB >
struct Add_Impl;
template < class FA, class FB >
struct Add_Alias;
template < class FA, class FB >
using Add = typename Add_Alias< FA, FB >::type;

template < class FA, class FB >
struct Scal_Impl;
template < class FA, class FB >
struct Scal_Alias;
template < class FA, class FB >
using Scal = typename Scal_Alias< FA, FB >::type;

//////////////////////////////////////////////////////////////
////               ADDITION : Add< FA,FB >                ////
//////////////////////////////////////////////////////////////


template < class FA, class FB >
struct Add_Impl : BinaryOp< Add_Impl, FA, FB > {
  // Output dim = FA::DIM = FB::DIM
  static const int DIM = FA::DIM;
  static_assert(DIM == FB::DIM, "Dimensions must be the same for Add");

  static void PrintIdString(::std::stringstream &str) {
    str << " + ";
  }

  static DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outA, __TYPE__ *outB) {
#pragma unroll
    for (int k = 0; k < DIM; k++)
      out[k] = outA[k] + outB[k];
  }

  // [ \partial_V (A + B) ] . gradin = [ \partial_V A ] . gradin  + [ \partial_V B ] . gradin
  template < class V, class GRADIN >
  using DiffT = Add< typename FA::template DiffT< V, GRADIN >, typename FB::template DiffT< V, GRADIN > >;

};

// Addition with scalar-> vector broadcasting on the left
template < class FA, class FB >
struct Add_Impl_Broadcast : BinaryOp< Add_Impl_Broadcast, FA, FB > {
  // Output dim = FB::DIM
  static const int DIM = FB::DIM;

  static void PrintIdString(::std::stringstream &str) {
    str << "+";
  }

  static DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outA, __TYPE__ *outB) {
#pragma unroll
    for (int k = 0; k < DIM; k++)
      out[k] = *outA + outB[k];
  }

  // [\partial_V (A + B) ] . gradin = [\partial_V A ] . gradin  + [\partial_V B ] . gradin
  template < class V, class GRADIN >
  using DiffT = Add< typename FA::template DiffT< V, Sum< GRADIN > >, typename FB::template DiffT< V, GRADIN > >;

};


// Simplification rules
// We have to divide rules into several stages
// to avoid conflicts

// third stage

// base class : this redirects to the implementation
template < class FA, class FB >
struct Add_Alias0 {
  using type1 = CondType< Add_Impl_Broadcast< FA, FB >, Add_Impl< FA, FB >, FA::DIM == 1 >;
  using type2 = CondType< Add_Impl_Broadcast< FB, FA >, type1, FB::DIM == 1 >;
  using type = CondType< Add_Impl< FA, FB >, type2, FA::DIM == FB::DIM >;
};

// A + A = 2A
template < class F >
struct Add_Alias0< F, F > {
  using type = Scal< IntConstant< 2 >, F >;
};

// A + B*A = (1+B)*A
template < class F, class G >
struct Add_Alias0< F, Scal_Impl< G, F>> {
  using type = Scal< Add< IntConstant< 1 >, G >, F >;
};

// B*A + A = (1+B)*A
template < class F, class G >
struct Add_Alias0< Scal_Impl< G, F >, F > {
  using type = Scal< Add< IntConstant< 1 >, G >, F >;
};

// second stage

// base class : this redirects to the third stage
template < class FA, class FB >
struct Add_Alias1 {
  using type = typename Add_Alias0< FA, FB >::type;
};

// B*A + C*A = (B+C)*A
template < class F, class G, class H >
struct Add_Alias1< Scal_Impl< G, F >, Scal_Impl< H, F> > {
  using type = Scal< Add< G, H >, F >;
};

// A+n = n+A (brings integers constants to the left)
template < int N, class F >
struct Add_Alias1< F, IntConstant_Impl< N> > {
  using type = Add< IntConstant< N >, F >;
};

// first stage

// base class : this redirects to the second stage
template < class FA, class FB >
struct Add_Alias {
  using type = typename Add_Alias1< FA, FB >::type;
};

// A + 0 = A
template < class FA, int DIM >
struct Add_Alias< FA, Zero< DIM>> {
  static_assert(DIM == FA::DIM, "Dimensions must be the same for Add");
  using type = FA;
};

// 0 + B = B
template < class FB, int DIM >
struct Add_Alias< Zero< DIM >, FB > {
  static_assert(DIM == FB::DIM, "Dimensions must be the same for Add");
  using type = FB;
};

// 0 + 0 = la tete a Toto
template < int DIM1, int DIM2 >
struct Add_Alias< Zero< DIM1 >, Zero< DIM2>> {
  static_assert(DIM1 == DIM2, "Dimensions must be the same for Add");
  using type = Zero< DIM1 >;
};

// m+n = m+n
template < int M, int N >
struct Add_Alias< IntConstant_Impl< M >, IntConstant_Impl< N > > {
  using type = IntConstant< M + N >;
};

template < class FA, class FB >
KeopsNS< Add< FA, FB > > operator+(KeopsNS< FA > fa, KeopsNS< FB > fb) {
  return KeopsNS< Add< FA, FB > >();
}

#define Add(fa, fb) KeopsNS<Add<decltype(InvKeopsNS(fa)),decltype(InvKeopsNS(fb))>>()

}
