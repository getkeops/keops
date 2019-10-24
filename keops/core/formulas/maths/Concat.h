#pragma once

#include <sstream>
#include "core/autodiff/BinaryOp.h"
#include "core/formulas/maths/Extract.h"
#include "core/formulas/maths/Add.h"
#include "core/pre_headers.h"

namespace keops {


//////////////////////////////////////////////////////////////
////     VECTOR CONCATENATION : Concat<F,G>               ////
//////////////////////////////////////////////////////////////

template<class F, class G>
struct Concat_Impl : BinaryOp<Concat_Impl, F, G> {
  static const int DIM = F::DIM + G::DIM;

  static void PrintId(::std::stringstream &str) {
    str << "Concat";
  }

  static DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outF, __TYPE__ *outG) {
#pragma unroll
    for (int k = 0; k < F::DIM; k++)
      out[k] = outF[k];
#pragma unroll
    for (int k = 0; k < G::DIM; k++)
      out[k + F::DIM] = outG[k];
  }

  template<class V, class GRADIN>
  using DiffTF = typename F::template DiffT<V, GRADIN>;

  template<class V, class GRADIN>
  using DiffTG = typename G::template DiffT<V, GRADIN>;

  template<class V, class GRADIN>
  using DiffT = Add<DiffTF<V, Extract<GRADIN, 0, F::DIM>>, DiffTG<V, Extract<GRADIN, F::DIM, G::DIM>>>;
};

template<class F, class G>
struct Concat_Alias {
  using type = Concat_Impl<F, G>;
};

// ugly stuff to make logsumexp reduction work
struct Dummy {
  static const int N = 0;
  static const int DIM = 0;
};

template<class F>
struct Concat_Alias<F, Dummy> {
  using type = F;
};

template<class F, class G>
using Concat = typename Concat_Alias<F, G>::type;

#define Concat(f,g) KeopsNS<Concat<decltype(InvKeopsNS(f)),decltype(InvKeopsNS(g))>>()

}
