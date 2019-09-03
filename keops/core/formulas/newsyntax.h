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
#include "core/formulas/maths/Add.h"
#include "core/formulas/maths/Concat.h"
#include "core/formulas/maths/Scal.h"
#include "core/formulas/maths/Mult.h"
#include "core/formulas/maths/ScalOrMult.h"
#include "core/formulas/maths/Subtract.h"
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
#include "core/formulas/maths/StandardBasis.h"
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

#include "core/reductions/sum.h"
#include "core/reductions/max_sumshiftexp.h"
#include "core/reductions/min.h"
#include "core/reductions/max.h"
#include "core/reductions/kmin.h"




namespace keops {

/*
 * This two dummy classes are used to prevent the compiler to be lost
 * during the resolution of the templated formula.
 */

template < class F > struct KeopsNS : public F { };

template < class F >
F InvKeopsNS(KeopsNS<F> kf) {
    return F();
}

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

#define Sum(p) KeopsNS<Sum<decltype(InvKeopsNS(p))>>()
#define SumT(p,d) KeopsNS<SumT<decltype(InvKeopsNS(p)),d>>()

#define Concat(f,g) KeopsNS<Concat<decltype(InvKeopsNS(f)),decltype(InvKeopsNS(g))>>()

// Formula compression

#define Factorize(F,G) KeopsNS<Factorize<decltype(InvKeopsNS(F)),decltype(InvKeopsNS(G))>()
#define AutoFactorize(F) KeopsNS<AutoFactorize<decltype(InvKeopsNS(F))>>()

// Differential operators for autograd

#define Grad(F,V,GRADIN)  KeopsNS<Grad<decltype(InvKeopsNS(F)),decltype(InvKeopsNS(V)),decltype(InvKeopsNS(GRADIN))>>()
#define Grad_WithSavedForward(F,V,GRADIN,FO)  KeopsNS<Grad_WithSavedForward<decltype(InvKeopsNS(F)),decltype(InvKeopsNS(V)),decltype(InvKeopsNS(GRADIN)),decltype(InvKeopsNS(FO))>>()
#define GradFromPos(F,V,I)  KeopsNS<GradFromPos<decltype(InvKeopsNS(F)),decltype(InvKeopsNS(V)),I>>()



////////////////////////////////////////////////////////////////////////////////
//                           Maths operations                                 //
////////////////////////////////////////////////////////////////////////////////

// Basic operators

template < class FA, class FB >
KeopsNS<Add<FA,FB>> operator+(KeopsNS<FA> fa, KeopsNS<FB> fb) {
    return KeopsNS<Add<FA,FB>>();
}
#define Add(fa,fb) KeopsNS<Add<decltype(InvKeopsNS(fa)),decltype(InvKeopsNS(fb))>>()

template < class FA, class FB >
KeopsNS<ScalOrMult<FA,FB>> operator*(KeopsNS<FA> fa, KeopsNS<FB> fb) {
    return KeopsNS<ScalOrMult<FA,FB>>();
}
#define ScalOrMult(fa,fb) KeopsNS<ScalOrMult<decltype(InvKeopsNS(fa)),decltype(InvKeopsNS(fb))>>()

template < class F >
KeopsNS<Minus<F>> operator-(KeopsNS<F> f) {
    return KeopsNS<Minus<F>>();
}
#define Minus(f) KeopsNS<Minus<decltype(InvKeopsNS(f))>>()

template < class FA, class FB >
KeopsNS<Subtract<FA,FB>> operator-(KeopsNS<FA> fa, KeopsNS<FB> fb) {
    return KeopsNS<Subtract<FA,FB>>();
}
#define Subtract(fa,fb) KeopsNS<Subtract<decltype(InvKeopsNS(fa)),decltype(InvKeopsNS(fb))>>()

template < class FA, class FB >
KeopsNS<Divide<FA,FB>> operator/(KeopsNS<FA> fa, KeopsNS<FB> fb) {
    return KeopsNS<Divide<FA,FB>>();
}
#define Divide(fa,fb) KeopsNS<Divide<decltype(InvKeopsNS(fa)),decltype(InvKeopsNS(fb))>>()

template < class FA, class FB >
KeopsNS<Scalprod<FA,FB>> operator|(KeopsNS<FA> fa, KeopsNS<FB> fb) {
    return KeopsNS<Scalprod<FA,FB>>();
}
#define Scalprod(fa,fb) KeopsNS<Scalprod<decltype(InvKeopsNS(fa)),decltype(InvKeopsNS(fb))>>()


// Basic functions

#define Abs(f) KeopsNS<Abs<decltype(InvKeopsNS(f))>>()
#define Exp(f) KeopsNS<Exp<decltype(InvKeopsNS(f))>>()
#define Cos(f) KeopsNS<Cos<decltype(InvKeopsNS(f))>>()
#define Sin(f) KeopsNS<Sin<decltype(InvKeopsNS(f))>>()
#define ReLU(f) KeopsNS<ReLU<decltype(InvKeopsNS(f))>>()
#define Step(f) KeopsNS<Step<decltype(InvKeopsNS(f))>>()
#define Sign(f) KeopsNS<Sign<decltype(InvKeopsNS(f))>>()
#define Pow(f,M) KeopsNS<Pow<decltype(InvKeopsNS(f)),M>>()
#define Square(f) KeopsNS<Square<decltype(InvKeopsNS(f))>>()
#define Inv(f) KeopsNS<Inv<decltype(InvKeopsNS(f))>>()
#define IntInv(N) KeopsNS<IntInv<N>>()
#define Log(f) KeopsNS<Log<decltype(InvKeopsNS(f))>>()
#define Powf(fa,fb) KeopsNS<Powf<decltype(InvKeopsNS(fa)),decltype(InvKeopsNS(fb))>>()
#define Sqrt(f) KeopsNS<Sqrt<decltype(InvKeopsNS(f))>>()
#define Rsqrt(f) KeopsNS<Rsqrt<decltype(InvKeopsNS(f))>>()

// Linear and tensor algebra

#define MatVecMult(f,g) KeopsNS<MatVecMult<decltype(InvKeopsNS(f)),decltype(InvKeopsNS(g))>>()
#define VecMatMult(f,g) KeopsNS<VecMatMult<decltype(InvKeopsNS(f)),decltype(InvKeopsNS(g))>>()
#define TensorProd(f,g) KeopsNS<TensorProd<decltype(InvKeopsNS(f)),decltype(InvKeopsNS(g))>>()
#define TensorDot(f,g,...) KeopsNS<TensorDot<decltype(InvKeopsNS(f)),decltype(InvKeopsNS(g)), __VA_ARGS__>>()

#define GradMatrix(f,g) KeopsNS<GradMatrix<decltype(InvKeopsNS(f)),decltype(InvKeopsNS(g))>>()

#define SqNorm2(f) KeopsNS<SqNorm2<decltype(InvKeopsNS(f))>>()
#define Norm2(f) KeopsNS<Norm2<decltype(InvKeopsNS(f))>>()
#define Normalize(f) KeopsNS<Normalize<decltype(InvKeopsNS(f))>>()
#define SqDist(f,g) KeopsNS<SqDist<decltype(InvKeopsNS(f)),decltype(InvKeopsNS(g))>>()
#define WeightedSqNorm(s,f)   KeopsNS<WeightedSqNorm<decltype(InvKeopsNS(s)), decltype(InvKeopsNS(f))>>()
#define WeightedSqDist(s,f,g) KeopsNS<WeightedSqDist<decltype(InvKeopsNS(s)), decltype(InvKeopsNS(f)),decltype(InvKeopsNS(g))>>()



////////////////////////////////////////////////////////////////////////////////
//                            Kernels                                         //
////////////////////////////////////////////////////////////////////////////////

#define GaussKernel(C,X,Y,B) KeopsNS<GaussKernel<decltype(InvKeopsNS(C)),decltype(InvKeopsNS(X)),decltype(InvKeopsNS(Y)),decltype(InvKeopsNS(B))>>()
#define CauchyKernel(C,X,Y,B) KeopsNS<CauchyKernel<decltype(InvKeopsNS(C)),decltype(InvKeopsNS(X)),decltype(InvKeopsNS(Y)),decltype(InvKeopsNS(B))>>()
#define LaplaceKernel(C,X,Y,B) KeopsNS<LaplaceKernel<decltype(InvKeopsNS(C)),decltype(InvKeopsNS(X)),decltype(InvKeopsNS(Y)),decltype(InvKeopsNS(B))>>()
#define InverseMultiquadricKernel(C,X,Y,B) KeopsNS<InverseMultiquadricKernel<decltype(InvKeopsNS(C)),decltype(InvKeopsNS(X)),decltype(InvKeopsNS(Y)),decltype(InvKeopsNS(B))>>()
#define SumGaussKernel(C,W,X,Y,B) KeopsNS<SumGaussKernel<decltype(InvKeopsNS(C)),decltype(InvKeopsNS(W)),decltype(InvKeopsNS(X)),decltype(InvKeopsNS(Y)),decltype(InvKeopsNS(B))>>()

#define DivFreeGaussKernel(C,X,Y,B) KeopsNS<DivFreeGaussKernel<decltype(InvKeopsNS(C)),decltype(InvKeopsNS(X)),decltype(InvKeopsNS(Y)),decltype(InvKeopsNS(B))>>()
#define CurlFreeGaussKernel(C,X,Y,B) KeopsNS<CurlFreeGaussKernel<decltype(InvKeopsNS(C)),decltype(InvKeopsNS(X)),decltype(InvKeopsNS(Y)),decltype(InvKeopsNS(B))>>()
#define TRIGaussKernel(L,C,X,Y,B) KeopsNS<TRIGaussKernel<decltype(InvKeopsNS(L)),decltype(InvKeopsNS(C)),decltype(InvKeopsNS(X)),decltype(InvKeopsNS(Y)),decltype(InvKeopsNS(B))>>()



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
