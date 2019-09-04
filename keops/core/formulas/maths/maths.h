#pragma once


/*
 * The file where the elementary math operators are defined.
 * Available math operations are :
 *
 *   +, *, - :
 *      Add<FA,FB>                  : adds FA and FB functions
 *      Subtract<FA,FB>             : subtractss FA and FB functions
 *      Scal<FA,FB>                 : product of FA (scalar valued) with FB
 *      Mult<FA,FB>                 : element-wise multiplication of FA and FB
 *      Minus<F>                    : alias for Scal<IntConstant<-1>,F>
 *
 *   /, ^, ^2, ^-1, ^(1/2) :
 *      Divide<FA,FB>               : alias for Scal<FA,Inv<FB>>
 *      Pow<F,M>                    : Mth power of F (vectorized) ; M is an integer
 *      Powf<A,B>                   : alias for Exp<Scal<FB,Log<FA>>>
 *      Square<F>                   : Pointwise square, more efficient than Pow<F,2>
 *      Inv<F>                      : Pointwise inverse, more efficient than Pow<F,-1>
 *      IntInv<N>                   : alias for Inv<IntConstant<N>>
 *      Sqrt<F>                     : alias for Powf<F,IntInv<2>>
 *      Rsqrt<F>                    : inverse square root
 *
 *   standard math functions :
 *      Sum<F>						: sum of values in F
 *      Abs<F>						: absolute value of F (vectorized)
 *      Exp<F>                      : exponential of F (vectorized)
 *      Log<F>                      : logarithm   of F (vectorized)
 *      Sin<F>                      : sine        of F (vectorized)
 *      Cos<F>                      : cosine      of F (vectorized)
 *      Sign<F>                     : sign        of F (vectorized)
 *      Step<F>                     : step        of F (vectorized)
 *      ReLU<F>                     : ReLU        of F (vectorized)
 *      Sign<F>                     : sign        of F (vectorized)
 *
 *   concatenation and matrix-vector products:
 *      Concat<FA,FB>               : concatenation of FB and FB
 *      MatVecMult<FA,FB>           : matrix-vector product (FA::DIM must be a muliple of FB::DIM)
 *      VecMatMult<FA,FB>           : vector-matrix product (FB::DIM must be a muliple of FA::DIM)
 *      TensorProd<FA,FB>           : tensor product (output is of dimension FA::DIM*FB::DIM)
 *      TensorDot<FA,FB, I>         : tensor dot as in numpy (FA::DIM must be a muliple of FB::DIM) I is an IntCst()
 *
 */

namespace keops {


// Addition, Subtraction, Scalar product and "Scalar*Vector product" symbolic operators.
// The actual implementation can be found below.
// Since the gradients of these operations are "bootstrapped", we need to be a little bit
// careful with the declaration order, and therefore use three "typenames" per operation:
// Op_Alias, Op_Impl and Op (proper).

template<class FA, class FB>
struct Add_Impl;
template<class FA, class FB>
struct Add_Alias;
template<class FA, class FB>
using Add = typename Add_Alias<FA, FB>::type;


template<class FA, class FB>
struct Subtract_Impl;
template<class FA, class FB>
struct Subtract_Alias;
template<class FA, class FB>
using Subtract = typename Subtract_Alias<FA, FB>::type;

template<class FA, class FB>
struct Scal_Impl;
template<class FA, class FB>
struct Scal_Alias;
template<class FA, class FB>
using Scal = typename Scal_Alias<FA, FB>::type;


template<class FA, class FB>
struct Mult_Impl;
template<class FA, class FB>
struct Mult_Alias;
template<class FA, class FB>
using Mult = typename Mult_Alias<FA, FB>::type;

/*
//template<class F>
//struct Sum_Impl;
template<class F>
struct Sum_Alias;
template<class F>
using Sum = typename Sum_Alias<F>::type;
*/

/*
template<class FA, class FB>
struct Scalprod_Impl;
template<class FA, class FB>
struct Scalprod_Alias;
template<class FA, class FB>
using Scalprod = typename Scalprod_Alias<FA, FB>::type;
*/


// template<class F>
// struct Norm2_Alias;
//template<class A, class B>
//struct TensorProd;


//template<class A, class B>
//struct MatVecMult;



/*
template<class F>
struct Minus_Impl;
template<class F>
using Minus = typename Minus_Impl<F>::type;

template<class F>
struct Sum_Impl;
template<class F>
using Sum = typename Sum_Impl<F>::type;

template<class FA, int I>
struct SumT_Impl;
template<class F, int I>
using SumT = typename SumT_Impl<F, I>::type;

template<class FA, class FB>
struct Add_Impl;
template<class FA, class FB>
struct Add_Alias;
template<class FA, class FB>
using Add = typename Add_Alias<FA, FB>::type;

template<class FA, class FB>
struct Concat_Impl;
template<class FA, class FB>
struct Concat_Alias;
template<class FA, class FB>
using Concat = typename Concat_Alias<FA, FB>::type;

template<class FA, class FB>
struct Scal_Impl;
template<class FA, class FB>
struct Scal_Alias;
template<class FA, class FB>
using Scal = typename Scal_Alias<FA, FB>::type;

template<class FA, class FB>
struct Mult_Impl;
template<class FA, class FB>
struct Mult_Alias;
template<class FA, class FB>
using Mult = typename Mult_Alias<FA, FB>::type;

template<class FA, class FB>
struct ScalOrMult_Impl;
template<class FA, class FB>
using ScalOrMult = typename ScalOrMult_Impl<FA, FB>::type;

template<class FA, class FB>
struct Subtract_Impl;
template<class FA, class FB>
struct Subtract_Alias;
template<class FA, class FB>
using Subtract = typename Subtract_Alias<FA, FB>::type;

template<class F>
struct Exp_Impl;
template<class F>
using Exp = typename Exp_Impl<F>::AllTypes  ;

template<class F>
struct Sin_Impl;
template<class F>
using Sin = typename Sin_Impl<F>::type;

template<class F>
struct Cos_Impl;
template<class F>
using Cos = typename Cos_Impl<F>::type;

template<class FA, int I>
struct Pow_Impl;
template<class FA, int I>
using Pow = typename Pow_Impl<FA, I>::type;

template<class F>
struct Square_Impl;
template<class F>
using Square = typename Square_Impl<F>::AllTypes ;

template<class F>
struct Inv_Impl;
template<class F>
using Inv = typename Inv_Impl<F>::type;

template<int F>
struct IntInv_Impl;
template<int F>
using IntInv = typename IntInv_Impl<F>::type;

template<class FA, class FB>
struct Divide_Impl;
template<class FA, class FB>
using Divide = typename Divide_Impl<FA, FB>::type;

template<class F>
struct Log_Impl;
template<class F>
using Log = typename Log_Impl<F>::type;

template<class F>
struct Sign_Impl;
template<class F>
using Sign = typename Sign_Impl<F>::type;

template<class F>
struct Abs_Impl;
template<class F>
using Abs = typename Abs_Impl<F>::type;

template<class F>
struct Step_Impl;
template<class F>
using Step = typename Step_Impl<F>::type;

template<class F>
struct ReLu_Impl;
template<class F>
using ReLu = typename ReLu_Impl<F>::type;

template<class FA, class FB>
struct Powf_Impl;
template<class FA, class FB>
using Powf = typename Powf_Impl<FA, FB>::type;

template<class F>
struct Sqrt_Impl;
template<class F>
struct Sqrt_Alias;
template<class F>
using Sqrt = typename Sqrt_Alias<F>::type;

template<class F>
struct Rsqrt_Impl;
template<class F>
struct Rsqrt_Alias;
template<class F>
using Rsqrt = typename Rsqrt_Alias<F>::type;

template<class FA, class FB>
struct MatVecMult_Impl;
template<class FA, class FB>
using MatVecMult = typename MatVecMult_Impl<FA, FB>::type;

template<class FA, class FB>
struct GradMatrix_Impl;
template<class FA, class FB>
using GradMatrix = typename GradMatrix_Impl<FA, FB>::type;


template<class FA, class FB, class DA, class DB, class CA, class CB, class P>
struct TensorDot_Impl;
template<class FA, class FB, class DA, class DB, class CA, class CB, class P>
using TensorDot = typename TensorDot_Impl<FA, FB, DA, DB, CA, CB, P>::type;

template<class FA, class FB>
struct TensorProd_Impl;
template<class FA, class FB>
using TensorProd = typename TensorProd_Impl<FA, FB>::type;

template<int DIM, int I>
struct StandardBasis_Impl;
template<int DIM, int I>
using StandardBasis = typename StandardBasis_Impl<DIM, I>::type;

template<class FA, class FB>
struct VecMatMult_Impl;
template<class FA, class FB>
using VecMatMult = typename VecMatMult_Impl<FA, FB>::type;
*/



}

/*
#include "core/formulas/maths/Minus.h"
#include "core/formulas/maths/Sum.h"
#include "core/formulas/maths/Add.h"
#include "core/formulas/maths/Concat.h"
#include "core/formulas/maths/Scal.h"
#include "core/formulas/maths/Mult.h"
#include "core/formulas/maths/ScalOrMult.h"
#include "core/formulas/maths/Substract.h"
#include "core/formulas/maths/Exp.h"
#include "core/formulas/maths/Sin.h"
#include "core/formulas/maths/Cos.h"
#include "core/formulas/maths/Pow.h"
#include "core/formulas/maths/Square.h"
#include "core/formulas/maths/Inv.h"
#include "core/formulas/maths/IntInv.h"
#include "core/formulas/maths/Divide.h"
#include "core/formulas/maths/Log.h"
#include "core/formulas/maths/Sign.h"
#include "core/formulas/maths/Abs.h"
#include "core/formulas/maths/Step.h"
#include "core/formulas/maths/ReLu.h"
#include "core/formulas/maths/Powf.h"
#include "core/formulas/maths/Sqrt.h"
#include "core/formulas/maths/Rsqrt.h"
#include "core/formulas/maths/MatVecMult.h"
#include "core/formulas/maths/GradMatrix.h"
#include "core/formulas/maths/TensorDot.h"
#include "core/formulas/maths/TensorProd.h"
#include "core/formulas/maths/VecMatMult.h"
*/


