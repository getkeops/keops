#pragma once

#include <sstream>
#include <assert.h>

#include "core/autodiff/UnaryOp.h"
#include "core/formulas/complex/ComplexReal.h"
#include "core/formulas/complex/ComplexImag.h"
#include "core/formulas/complex/Real2Complex.h"
#include "core/formulas/complex/Imag2Complex.h"
#include "core/formulas/maths/Add.h"
#include "core/formulas/maths/Mult.h"
#include "core/formulas/maths/Exp.h"
#include "core/formulas/maths/Cos.h"
#include "core/formulas/maths/Sin.h"

#include "core/utils/keops_math.h"
#include "core/pre_headers.h"

namespace keops {


/////////////////////////////////////////////////////////////////////////
////      ComplexExp                           ////
/////////////////////////////////////////////////////////////////////////

template<class F>
struct ComplexExp: UnaryOp<ComplexExp, F> {

  static_assert(F::DIM % 2 == 0, "Dimension of F must be even");

  static const int DIM = F::DIM;

  static void PrintIdString(::std::stringstream &str) { str << "ComplexExp"; }
    
  template < typename TYPE > 
  static DEVICE INLINE void Operation(TYPE *out, TYPE *inF) {
        #pragma unroll
        for (int i = 0; i < F::DIM; i+=2) {
            TYPE r = keops_exp(inF[i]);
            out[i] =  r * keops_cos(inF[i+1]);
            out[i+1] = r * keops_sin(inF[i+1]);
        }
    }
    
    // building equivalent formula for autodiff
    using AltAbs = Exp<ComplexReal<F>>;
    using AltReal = Mult < AltAbs, Cos<ComplexImag<F>> >;
    using AltImag = Mult < AltAbs, Sin<ComplexImag<F>> >;
    using AltFormula = Add < Real2Complex<AltReal> , Imag2Complex<AltImag> >;

    template<class V, class GRADIN>
    using DiffT = typename AltFormula::template DiffT<V, GRADIN>;

};

#define ComplexExp(f) KeopsNS<ComplexExp<decltype(InvKeopsNS(f))>>()

}
