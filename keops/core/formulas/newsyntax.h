/*
 * This file contains compilers macros (mostly aliases) used to defined
 * high end user friendly formulas.
 *
 */

#pragma once

#include "core/formulas/constants.h"
#include "core/formulas/factorize.h"

#include "core/formulas/maths/Minus.h"
#include "core/formulas/maths/Sum.h"
#include "core/formulas/maths/Concat.h"
#include "core/formulas/maths/Scal.h"
#include "core/formulas/maths/Mult.h"
#include "core/formulas/maths/ScalOrMult.h"
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

// import all operation on vector implementations
#include "core/formulas/norms/Norm2.h"
#include "core/formulas/norms/Normalize.h"
#include "core/formulas/norms/Scalprod.h"
#include "core/formulas/norms/SqDist.h"
#include "core/formulas/norms/SqNorm2.h"
#include "core/formulas/norms/SqNormDiag.h"
#include "core/formulas/norms/SqNormIso.h"
#include "core/formulas/norms/WeightedSqDist.h"
#include "core/formulas/norms/WeightedSqNorm.h"

// import all Kernels
#include "core/formulas/kernels/ScalarRadialKernels.h"

// import all reductions
#include "core/reductions/sum.h"
#include "core/reductions/max_sumshiftexp.h"
#include "core/reductions/min.h"
#include "core/reductions/max.h"
#include "core/reductions/kmin.h"

#include "core/pre_headers.h"


namespace keops {



////////////////////////////////////////////////////////////////////////////////
//                           FORMULAS                                         //
////////////////////////////////////////////////////////////////////////////////

// Variables

#define Var(N,DIM, CAT) KeopsNS<Var<N,DIM,CAT>>()

#define Vi(N,DIM) KeopsNS<_X<N,DIM>>()
#define Vj(N,DIM) KeopsNS<_Y<N,DIM>>()
#define Pm(N,DIM) KeopsNS<_P<N,DIM>>()
#define Ind(...) index_sequence<__VA_ARGS__>

#define IntCst(N) KeopsNS<IntConstant<N>>()
#define Zero(D) KeopsNS<Zero<D>>()

#define Elem(p,k) KeopsNS<Elem<decltype(InvKeopsNS(p)),k>>()
#define ElemT(p,k) KeopsNS<ElemT<decltype(InvKeopsNS(p)),k>>()

#define Extract(p,k,n) KeopsNS<Extract<decltype(InvKeopsNS(p)),k,n>>()
#define ExtractT(p,k,n) KeopsNS<ExtractT<decltype(InvKeopsNS(p)),k,n>>()



#define Concat(f,g) KeopsNS<Concat<decltype(InvKeopsNS(f)),decltype(InvKeopsNS(g))>>()

// Formula compression



// Differential operators for autograd

#define Grad(F,V,GRADIN)  KeopsNS<Grad<decltype(InvKeopsNS(F)),decltype(InvKeopsNS(V)),decltype(InvKeopsNS(GRADIN))>>()
#define Grad_WithSavedForward(F,V,GRADIN,FO)  KeopsNS<Grad_WithSavedForward<decltype(InvKeopsNS(F)),decltype(InvKeopsNS(V)),decltype(InvKeopsNS(GRADIN)),decltype(InvKeopsNS(FO))>>()
#define GradFromPos(F,V,I)  KeopsNS<GradFromPos<decltype(InvKeopsNS(F)),decltype(InvKeopsNS(V)),I>>()


////////////////////////////////////////////////////////////////////////////////
//                            Kernels                                         //
////////////////////////////////////////////////////////////////////////////////





////////////////////////////////////////////////////////////////////////////////
//                          reductions                                        //
////////////////////////////////////////////////////////////////////////////////

#define Sum_Reduction(F,I) KeopsNS<Sum_Reduction<decltype(InvKeopsNS(F)),I>>()
#define Max_SumShiftExp_Reduction(F,I) KeopsNS<Max_SumShiftExp_Reduction<decltype(InvKeopsNS(F)),I>>()
#define Max_SumShiftExpWeight_Reduction(F,I,G) KeopsNS<Max_SumShiftExp_Reduction<decltype(InvKeopsNS(F)),I,decltype(InvKeopsNS(G))>>()

#define Min_ArgMin_Reduction(F,I) KeopsNS<Min_ArgMin_Reduction<decltype(InvKeopsNS(F)),I>>()
#define ArgMin_Reduction(F,I) KeopsNS<ArgMin_Reduction<decltype(InvKeopsNS(F)),I>>()
#define Min_Reduction(F,I) KeopsNS<Min_Reduction<decltype(InvKeopsNS(F)),I>>()

#define Max_ArgMax_Reduction(F,I) KeopsNS<Max_ArgMax_Reduction<decltype(InvKeopsNS(F)),I>>()
#define ArgMax_Reduction(F,I) KeopsNS<ArgMax_Reduction<decltype(InvKeopsNS(F)),I>>()
#define Max_Reduction(F,I) KeopsNS<Max_Reduction<decltype(InvKeopsNS(F)),I>>()

#define KMin_ArgKMin_Reduction(F,K,I) KeopsNS<KMin_ArgKMin_Reduction<decltype(InvKeopsNS(F)),K,I>>()
#define ArgKMin_Reduction(F,K,I) KeopsNS<ArgKMin_Reduction<decltype(InvKeopsNS(F)),K,I>>()
#define KMin_Reduction(F,K,I) KeopsNS<KMin_Reduction<decltype(InvKeopsNS(F)),K,I>>()






////////////////////////////////////////////////////////////////////////////////
//         Deprecated : (old syntax, kept for backward compatibility)         //
////////////////////////////////////////////////////////////////////////////////

#define SumReduction(F,I) KeopsNS<Sum_Reduction<decltype(InvKeopsNS(F)),I>>()
#define LogSumExpReduction(F,I) KeopsNS<Max_SumShiftExp_Reduction<decltype(InvKeopsNS(F)),I>>()
#define LogSumExpVectReduction(F,I,G) KeopsNS<Max_SumShiftExp_Reduction<decltype(InvKeopsNS(F)),I,decltype(InvKeopsNS(G))>>()

#define Min_ArgMin_Reduction(F,I) KeopsNS<Min_ArgMin_Reduction<decltype(InvKeopsNS(F)),I>>()
#define ArgMinReduction(F,I) KeopsNS<ArgMin_Reduction<decltype(InvKeopsNS(F)),I>>()
#define MinReduction(F,I) KeopsNS<Min_Reduction<decltype(InvKeopsNS(F)),I>>()

#define MaxArgMaxReduction(F,I) KeopsNS<Max_ArgMax_Reduction<decltype(InvKeopsNS(F)),I>>()
#define ArgMaxReduction(F,I) KeopsNS<ArgMax_Reduction<decltype(InvKeopsNS(F)),I>>()
#define MaxReduction(F,I) KeopsNS<Max_Reduction<decltype(InvKeopsNS(F)),I>>()

#define KMinArgKMinReduction(F,K,I) KeopsNS<KMin_ArgKMin_Reduction<decltype(InvKeopsNS(F)),K,I>>()
#define ArgKMinReduction(F,K,I) KeopsNS<ArgKMin_Reduction<decltype(InvKeopsNS(F)),K,I>>()
#define KMinReduction(F,K,I) KeopsNS<KMin_Reduction<decltype(InvKeopsNS(F)),K,I>>()

#define Vx(N,DIM) KeopsNS<_X<N,DIM>>()
#define Vy(N,DIM) KeopsNS<_Y<N,DIM>>()

}
