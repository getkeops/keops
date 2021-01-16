#pragma once

#include <sstream>
#include <assert.h>

#include "core/autodiff/UnaryOp.h"
#include "core/formulas/complex/Real2Complex.h"
#include "core/formulas/complex/Imag2Complex.h"
#include "core/formulas/maths/Add.h"
#include "core/formulas/maths/Cos.h"
#include "core/formulas/maths/Sin.h"

#include "core/utils/keops_math.h"
#include "core/pre_headers.h"

namespace keops {


/////////////////////////////////////////////////////////////////////////
////      ComplexExp1j                           ////
/////////////////////////////////////////////////////////////////////////

template<class F>
struct ComplexExp1j: UnaryOp<ComplexExp1j, F> {

  static const int DIM = 2*F::DIM;

  static void PrintIdString(::std::stringstream &str) { str << "ComplexExp1j"; }
    
  template < typename TYPE > 
  static DEVICE INLINE void Operation(TYPE *out, TYPE *inF) {
        #pragma unroll
        for (int i = 0; i < F::DIM; i+=2) {
            keops_sincos(inF[i/2], out+i+1, out+i);
            //out[i] =  keops_cos(inF[i/2]);
            //out[i+1] = keops_sin(inF[i/2]);
        }
    }
    
    // building equivalent formula for autodiff
    using AltFormula = Add < Real2Complex<Cos<F>> , Imag2Complex<Sin<F>> >;

    template<class V, class GRADIN>
    using DiffT = typename AltFormula::template DiffT<V, GRADIN>;

};

#define ComplexExp1j(f) KeopsNS<ComplexExp1j<decltype(InvKeopsNS(f))>>()

}
