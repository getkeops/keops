#pragma once

#include <assert.h>
#include <sstream>

#include "core/autodiff/BinaryOp.h"
#include "core/formulas/maths/Scal.h"
#include "core/formulas/maths/Add.h"
#include "core/formulas/constants/Zero.h"

#include "core/pre_headers.h"

namespace keops {

template < class FA, class FB >
struct Scalprod_Alias;
template < class FA, class FB >
using Scalprod = typename Scalprod_Alias< FA, FB >::type;


template < class FA, class FB >
struct Scal_Impl;
template < class FA, class FB >
struct Scal_Alias;
template < class FA, class FB >
using Scal = typename Scal_Alias< FA, FB >::type;

template < class FA, class FB >
struct Add_Impl;
template < class FA, class FB >
struct Add_Alias;
template < class FA, class FB >
using Add = typename Add_Alias< FA, FB >::type;

//////////////////////////////////////////////////////////////
////           SCALAR PRODUCT :   Scalprod< A,B >         ////
//////////////////////////////////////////////////////////////


template < class FA, class FB >
struct Scalprod_Impl : BinaryOp< Scalprod_Impl, FA, FB > {
  // Output dimension = 1, provided that FA::DIM = FB::DIM
  static const int DIMIN = FA::DIM;
  static_assert(DIMIN == FB::DIM, "Dimensions must be the same for Scalprod");
  static const int DIM = 1;

  static void PrintIdString(::std::stringstream &str) { str << "|"; }

  static DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outA, __TYPE__ *outB) {
    *out = 0;
    for (int k = 0; k < DIMIN; k++)
      *out += outA[k] * outB[k];
  }

  // <A,B> is scalar-valued, so that gradin is necessarily a scalar.
  // [\partial_V <A,B>].gradin = gradin * ( [\partial_V A].B + [\partial_V B].A )
  template < class V, class GRADIN >
  using DiffT = Scal< GRADIN, Add< typename FA::template DiffT< V, FB >, typename FB::template DiffT< V, FA > > >;
};

template < class FA, class FB >
struct Scalprod_Alias {
  using type = Scalprod_Impl< FA, FB >;
};


// Three simple optimizations :

// <A,0> = 0
template < class FA, int DIM >
struct Scalprod_Alias< FA, Zero< DIM > > {
static_assert(DIM == FA::DIM, "Dimensions must be the same for Scalprod");
using type = Zero< 1 >;
};

// <0,B> = 0
template < class FB, int DIM >
struct Scalprod_Alias< Zero< DIM >, FB > {
static_assert(DIM == FB::DIM, "Dimensions must be the same for Scalprod");
using type = Zero< 1 >;
};

// <0,0> = 0
template < int DIM1, int DIM2 >
struct Scalprod_Alias< Zero< DIM1 >, Zero< DIM2 > > {
static_assert(DIM1 == DIM2, "Dimensions must be the same for Scalprod");
using type = Zero< 1 >;
};

template < class FA, class FB >
KeopsNS< Scalprod< FA, FB> > operator|(KeopsNS< FA > fa, KeopsNS< FB > fb) {
  return KeopsNS< Scalprod< FA, FB > > ();
}

#define Scalprod(fa, fb) KeopsNS< Scalprod< decltype(InvKeopsNS(fa)), decltype(InvKeopsNS(fb)) > >()

}
