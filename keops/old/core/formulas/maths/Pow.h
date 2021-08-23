#pragma once

#include <sstream>

#include "core/utils/keops_math.h"
#include "core/autodiff/VectorizedScalarUnaryOp.h"
#include "core/formulas/constants/IntConst.h"
#include "core/formulas/maths/Mult.h"
#include "core/formulas/maths/Scal.h"
#include "core/formulas/maths/Inv.h"
#include "core/formulas/maths/Powf.h"

namespace keops {

//////////////////////////////////////////////////////////////
////             POWER OPERATOR : Pow< F, M >             ////
//////////////////////////////////////////////////////////////

#if USE_HALF

// there is no power operator available yet for half2...
// so we use the recursive definition F^M = F^(M/2) * F^(M-M/2).
template<class F, int M>
struct Pow_Impl {
  using left = typename Pow_Impl<F,M/2>::type;
  using right = typename Pow_Impl<F,M-M/2>::type;
  using type = typename Mult_Alias<left, right>::type;
};

template<class F>
struct Pow_Impl<F,1> {
  using type = F;
};

template<class F>
struct Pow_Impl<F,-1> {
  using type = Inv<F>;
};

template<class F>
struct Pow_Impl<F,0> {
  using type = IntConstant<1>;
};

template<class F, int M>
using Pow = typename Pow_Impl<F,M>::type;

#else

template<class F, int M>
struct Pow : VectorizedScalarUnaryOp<Pow, F, M> {

  static void PrintIdString(::std::stringstream &str) { str << "Pow"; }


  template < typename TYPE > 
  struct Operation_Scalar {
	DEVICE INLINE void operator() (TYPE &out, TYPE &outF) {
    	  out = keops_pow(outF,M);
    }
  };

  // [\partial_V F^M].gradin  =  M * (F^(M-1)) * [\partial_V F].gradin
  template<class V, class GRADIN>
  using DiffTF = typename F::template DiffT<V, GRADIN>;

  template<class V, class GRADIN>
  using DiffT = Scal<IntConstant<M>, DiffTF<V, Mult<Pow<F, M - 1>, GRADIN>>>;

};

#endif

#define Pow(f,M) KeopsNS<Pow<decltype(InvKeopsNS(f)),M>>()

}
