#pragma once

#include <sstream>

#include "core/autodiff/BinaryOp.h"
#include "core/formulas/constants/Zero.h"
#include "core/formulas/maths/Mult.h"
#include "core/formulas/maths/Add.h"

namespace keops {

// We need some pre-declarations
template<class FA, class FB>
struct Mult_Impl;
template<class FA, class FB>
struct Mult_Alias;
template<class FA, class FB>
using Mult = typename Mult_Alias<FA, FB>::type;

//////////////////////////////////////////////////////////////
////      Element-wise Multiplication : Mult< FA,FB>      ////
//////////////////////////////////////////////////////////////


template<class FA, class FB>
struct Mult_Impl : BinaryOp<Mult_Impl, FA, FB> {
  // FA and FB are vectors with same size, Output has the same size
  static const int DIM = FA::DIM;
  static_assert(FA::DIM == DIM, "Dimensions of FA and FB must be the same for Mult");

  static void PrintIdString(::std::stringstream &str) {
    str << "*";
  }

  static DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outA, __TYPE__ *outB) {
#pragma unroll
    for (int k = 0; k < DIM; k++)
      out[k] = outA[k] * outB[k];
  }

  //  \diff_V (A*B) = (\diff_V A) * B + A * (\diff_V B)
  template<class V, class GRADIN>
  using DiffTFA = typename FA::template DiffT<V, GRADIN>;

  template<class V, class GRADIN>
  using DiffTFB = typename FB::template DiffT<V, GRADIN>;

  template<class V, class GRADIN>
  using DiffT = Add<DiffTFA<V, Mult<FB, GRADIN>>, DiffTFB<V, Mult<FA, GRADIN>>>;

};

// Simplification rules

// base class : this redirects to the implementation
template<class FA, class FB>
struct Mult_Alias {
  using type = Mult_Impl<FA, FB>;
};

// A * 0 = 0
template<class FA, int DIM>
struct Mult_Alias<FA, Zero<DIM>> {
static_assert(DIM == FA::DIM, "Dimensions of FA and FB must be the same for Mult");
using type = Zero<DIM>;
};

// 0 * B = 0
template<class FB, int DIM>
struct Mult_Alias<Zero<DIM>, FB> {
static_assert(DIM == FB::DIM, "Dimensions of FA and FB must be the same for Mult");
using type = Zero<DIM>;
};

// 0 * 0 = 0 (we have to specify it otherwise there is a conflict between A*0 and 0*B)
template<int DIM1, int DIM2>
struct Mult_Alias<Zero<DIM1>, Zero<DIM2>> {
static_assert(DIM1 == DIM2, "Dimensions of FA and FB must be the same for Mult");
using type = Zero<DIM1>;
};

}
