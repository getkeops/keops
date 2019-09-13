#pragma once

#include <sstream>

#include "core/autodiff/BinaryOp.h"
#include "core/formulas/constants/Zero.h"
#include "core/formulas/constants/IntConst.h"
#include "core/formulas/maths/Add.h"
#include "core/formulas/norms/Scalprod.h"


namespace keops {

// We need some pre-declarations due to the co-dependency of Add and Scal
template<class FA, class FB>
struct Scal_Impl;
template<class FA, class FB>
struct Scal_Alias;
template<class FA, class FB>
using Scal = typename Scal_Alias<FA, FB>::type;

template<class FA, class FB>
struct Add_Impl;
template<class FA, class FB>
struct Add_Alias;
template<class FA, class FB>
using Add = typename Add_Alias<FA, FB>::type;

//////////////////////////////////////////////////////////////
////      Scal*Vector Multiplication : Scal< FA,FB>       ////
//////////////////////////////////////////////////////////////


template<class FA, class FB>
struct Scal_Impl : BinaryOp<Scal_Impl, FA, FB> {
  // FB is a vector, Output has the same size, and FA is a scalar
  static const int DIM = FB::DIM;
  static_assert(FA::DIM == 1, "Dimension of FA must be 1 for Scal");

  static void PrintIdString(::std::stringstream &str) {
    str << "*";
  }

  static DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outA, __TYPE__ *outB) {
#pragma unroll
    for (int k = 0; k < DIM; k++)
      out[k] = *outA * outB[k];
  }

  //  \diff_V (A*B) = (\diff_V A) * B + A * (\diff_V B)
  // i.e.
  //  < \diff_V (A*B) . dV, gradin > = (\diff_V A).dV * <B,gradin> + A * < (\diff_V B).dV, gradin >
  //
  // so its L2 conjugate is given by :
  //
  // [\partial_V A*B] . gradin = [\partial_V A].(<gradin,B>) + A * [\partial_V B].gradin
  template<class V, class GRADIN>
  using DiffT = Add<typename FA::template DiffT<V, Scalprod<GRADIN, FB>>,
                    Scal<FA, typename FB::template DiffT<V, GRADIN> > >;

};

// Simplification rules
// We have to divide rules into several stages
// to avoid conflicts

// second stage

// base class : this redirects to the implementation
template<class FA, class FB>
struct Scal_Alias0 {
  using type = Scal_Impl<FA, FB>;
};

// a*(b*c) = (a*b)*c
template<class FA, class F, class G>
struct Scal_Alias0<FA, Scal_Impl<F, G>> {
  using type = Scal<Scal<FA, F>, G>;
};

// m*n = m*n
template<int M, int N>
struct Scal_Alias0<IntConstant_Impl<M>, IntConstant_Impl<N>> {
using type = IntConstant<M * N>;
};

// a*n = n*a
template<class FA, int N>
struct Scal_Alias0<FA, IntConstant_Impl<N>> {
using type = Scal<IntConstant<N>, FA>;
};

// first stage

// base class : this redirects to the second stage
template<class FA, class FB>
struct Scal_Alias {
  using type = typename Scal_Alias0<FA, FB>::type;
};

// A * 0 = 0
template<class FA, int DIM>
struct Scal_Alias<FA, Zero<DIM>> {
static_assert(1 == FA::DIM, "Dimension of FA must be 1 for Scal");
using type = Zero<DIM>;
};

// 0 * B = 0
template<class FB, int DIM>
struct Scal_Alias<Zero<DIM>, FB> {
static_assert(DIM == 1, "Dimension of FA must be 1 for Scal");
using type = Zero<FB::DIM>;
};

// 0 * 0 = 0 (we have to specify it otherwise there is a conflict between A*0 and 0*B)
template<int DIM1, int DIM2>
struct Scal_Alias<Zero<DIM1>, Zero<DIM2>> {
static_assert(DIM1 == 1, "Dimension of FA must be 1 for Scal");
using type = Zero<DIM2>;
};

}
