#pragma once

#include "core/utils/keops_math.h"
#include "core/autodiff/VectorizedScalarUnaryOp.h"
#include "core/formulas/maths/Mult.h"
#include "core/formulas/maths/Inv.h"
#include "core/formulas/maths/Add.h"
#include "core/formulas/constants/IntConst.h"
#include "core/formulas/maths/Square.h"


namespace keops {

//////////////////////////////////////////////////////////////
////                ARCTANGENT :  Atan< F >               ////
//////////////////////////////////////////////////////////////

template <class F>
struct Atan : VectorizedScalarUnaryOp<Atan, F>
{

    static void PrintIdString(::std::stringstream &str) { str << "Atan"; }

    template <typename TYPE>
    struct Operation_Scalar
    {
        DEVICE INLINE void operator()(TYPE &out, TYPE &outF)
        {
            out = keops_atan(outF);
        }
    };

    // dx = 1/(1 + x^2)
    template <class V, class GRADIN>
    using DiffT = typename F::template DiffT<V, Mult<Inv<Add<IntConstant<1>, Square<F>>>, GRADIN>>;

};

#define Atan(f) KeopsNS<Atan<decltype(InvKeopsNS(f))>>()

}
