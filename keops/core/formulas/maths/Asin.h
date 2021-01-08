#pragma once

#include "core/utils/keops_math.h"
#include "core/autodiff/VectorizedScalarUnaryOp.h"
#include "core/formulas/maths/Mult.h"
#include "core/formulas/maths/Inv.h"
#include "core/formulas/maths/Sqrt.h"
#include "core/formulas/maths/Subtract.h"
#include "core/formulas/constants/IntConst.h"
#include "core/formulas/maths/Square.h"


namespace keops {

//////////////////////////////////////////////////////////////
////                ARCSINE :  Asin< F >                ////
//////////////////////////////////////////////////////////////

template <class F>
struct Asin : VectorizedScalarUnaryOp<Asin, F>
{

    static void PrintIdString(::std::stringstream &str) { str << "Asin"; }

    template <typename TYPE>
    struct Operation_Scalar
    {
        DEVICE INLINE void operator()(TYPE &out, TYPE &outF)
        {
            out = keops_asin(outF);
        }
    };

    // dx = 1/sqrt(1 - x^2)
    template <class V, class GRADIN>
    using DiffT = typename F::template DiffT<V, Mult<Inv<Sqrt<Subtract<IntConstant<1>, Square<F>>>>, GRADIN>>;

};

#define Asin(f) KeopsNS<Asin<decltype(InvKeopsNS(f))>>()

}
