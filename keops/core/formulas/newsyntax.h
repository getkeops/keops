#pragma once

#include "core/formulas/constants.h"
#include "core/formulas/maths.h"
#include "core/formulas/kernels.h"
#include "core/formulas/norms.h"
#include "core/formulas/factorize.h"

#include "core/reductions/sum.h"
#include "core/reductions/log_sum_exp.h"
#include "core/reductions/min.h"
#include "core/reductions/kmin.h"

namespace keops {

template < class F > struct KeopsNS : public F { };

template < class F >
F InvKeopsNS(KeopsNS<F> kf) {
	return F();
}

#define Var(N,DIM, CAT) KeopsNS<Var<N,DIM,CAT>>()

#define Vx(N,DIM) KeopsNS<_X<N,DIM>>()

#define Vy(N,DIM) KeopsNS<_Y<N,DIM>>()

#define Pm(N,DIM) KeopsNS<_P<N,DIM>>()

#define Factorize(F,G) KeopsNS<Factorize<decltype(InvKeopsNS(F)),decltype(InvKeopsNS(G))>()

#define AutoFactorize(F) KeopsNS<AutoFactorize<decltype(InvKeopsNS(F))>>()

#define Grad(F,V,GRADIN)  KeopsNS<Grad<decltype(InvKeopsNS(F)),decltype(InvKeopsNS(V)),decltype(InvKeopsNS(GRADIN))>>()

#define Grad_WithSavedForward(F,V,GRADIN,FO)  KeopsNS<Grad_WithSavedForward<decltype(InvKeopsNS(F)),decltype(InvKeopsNS(V)),decltype(InvKeopsNS(GRADIN)),decltype(InvKeopsNS(FO))>>()

#define GradFromPos(F,V,I)  KeopsNS<GradFromPos<decltype(InvKeopsNS(F)),decltype(InvKeopsNS(V)),I>>()

#define IntCst(N) KeopsNS<IntConstant<N>>()

#define Elem(p,k) KeopsNS<Elem<decltype(InvKeopsNS(p)),k>>()

#define ElemT(p,k) KeopsNS<ElemT<decltype(InvKeopsNS(p)),k>>()

#define Extract(p,k,n) KeopsNS<Extract<decltype(InvKeopsNS(p)),k,n>>()

#define ExtractT(p,k,n) KeopsNS<ExtractT<decltype(InvKeopsNS(p)),k,n>>()

#define Concat(f,g) KeopsNS<Concat<decltype(InvKeopsNS(f)),decltype(InvKeopsNS(g))>>()

#define Exp(f) KeopsNS<Exp<decltype(InvKeopsNS(f))>>()

#define Pow(f,M) KeopsNS<Pow<decltype(InvKeopsNS(f)),M>>()

#define Square(f) KeopsNS<Square<decltype(InvKeopsNS(f))>>()

#define Inv(f) KeopsNS<Inv<decltype(InvKeopsNS(f))>>()

#define IntInv(N) KeopsNS<IntInv<N>>()

#define Log(f) KeopsNS<Log<decltype(InvKeopsNS(f))>>()

#define Powf(fa,fb) KeopsNS<Powf<decltype(InvKeopsNS(fa)),decltype(InvKeopsNS(fb))>>()

#define Sqrt(f) KeopsNS<Sqrt<decltype(InvKeopsNS(f))>>()

#define Rsqrt(f) KeopsNS<Rsqrt<decltype(InvKeopsNS(f))>>()

#define MatVecMult(f,g) KeopsNS<MatVecMult<decltype(InvKeopsNS(f)),decltype(InvKeopsNS(g))>>()

#define VecMatMult(f,g) KeopsNS<VecMatMult<decltype(InvKeopsNS(f)),decltype(InvKeopsNS(g))>>()

#define TensorProd(f,g) KeopsNS<TensorProd<decltype(InvKeopsNS(f)),decltype(InvKeopsNS(g))>>()

#define GradMatrix(f,v) KeopsNS<GradMatrix<decltype(InvKeopsNS(f)),decltype(InvKeopsNS(g))>>()

#define SqNorm2(f) KeopsNS<SqNorm2<decltype(InvKeopsNS(f))>>()

#define Norm2(f) KeopsNS<Norm2<decltype(InvKeopsNS(f))>>()

#define Normalize(f) KeopsNS<Normalize<decltype(InvKeopsNS(f))>>()

#define SqDist(f,g) KeopsNS<SqDist<decltype(InvKeopsNS(f)),decltype(InvKeopsNS(g))>>()

#define WeightedSqNorm(s,f)   KeopsNS<WeightedSqNorm<decltype(InvKeopsNS(s)), decltype(InvKeopsNS(f))>>()
#define WeightedSqDist(s,f,g) KeopsNS<WeightedSqDist<decltype(InvKeopsNS(s)), decltype(InvKeopsNS(f)),decltype(InvKeopsNS(g))>>()



#define GaussKernel(C,X,Y,B) KeopsNS<GaussKernel<decltype(InvKeopsNS(C)),decltype(InvKeopsNS(X)),decltype(InvKeopsNS(Y)),decltype(InvKeopsNS(B))>>()
#define CauchyKernel(C,X,Y,B) KeopsNS<CauchyKernel<decltype(InvKeopsNS(C)),decltype(InvKeopsNS(X)),decltype(InvKeopsNS(Y)),decltype(InvKeopsNS(B))>>()
#define LaplaceKernel(C,X,Y,B) KeopsNS<LaplaceKernel<decltype(InvKeopsNS(C)),decltype(InvKeopsNS(X)),decltype(InvKeopsNS(Y)),decltype(InvKeopsNS(B))>>()
#define InverseMultiquadricKernel(C,X,Y,B) KeopsNS<InverseMultiquadricKernel<decltype(InvKeopsNS(C)),decltype(InvKeopsNS(X)),decltype(InvKeopsNS(Y)),decltype(InvKeopsNS(B))>>()
#define SumGaussKernel(C,W,X,Y,B) KeopsNS<SumGaussKernel<decltype(InvKeopsNS(C)),decltype(InvKeopsNS(W)),decltype(InvKeopsNS(X)),decltype(InvKeopsNS(Y)),decltype(InvKeopsNS(B))>>()

#define DivFreeGaussKernel(C,X,Y,B) KeopsNS<DivFreeGaussKernel<decltype(InvKeopsNS(C)),decltype(InvKeopsNS(X)),decltype(InvKeopsNS(Y)),decltype(InvKeopsNS(B))>>()
#define CurlFreeGaussKernel(C,X,Y,B) KeopsNS<CurlFreeGaussKernel<decltype(InvKeopsNS(C)),decltype(InvKeopsNS(X)),decltype(InvKeopsNS(Y)),decltype(InvKeopsNS(B))>>()
#define TRIGaussKernel(L,C,X,Y,B) KeopsNS<TRIGaussKernel<decltype(InvKeopsNS(L)),decltype(InvKeopsNS(C)),decltype(InvKeopsNS(X)),decltype(InvKeopsNS(Y)),decltype(InvKeopsNS(B))>>()


template < class FA, class FB >
KeopsNS<Add<FA,FB>> operator+(KeopsNS<FA> fa, KeopsNS<FB> fb) {
    return KeopsNS<Add<FA,FB>>();
}

template < class FA, class FB >
KeopsNS<ScalOrMult<FA,FB>> operator*(KeopsNS<FA> fa, KeopsNS<FB> fb) {
    return KeopsNS<ScalOrMult<FA,FB>>();
}

template < class F >
KeopsNS<Minus<F>> operator-(KeopsNS<F> f) {
    return KeopsNS<Minus<F>>();
}

template < class FA, class FB >
KeopsNS<Subtract<FA,FB>> operator-(KeopsNS<FA> fa, KeopsNS<FB> fb) {
    return KeopsNS<Subtract<FA,FB>>();
}

template < class FA, class FB >
KeopsNS<Divide<FA,FB>> operator/(KeopsNS<FA> fa, KeopsNS<FB> fb) {
    return KeopsNS<Divide<FA,FB>>();
}

template < class FA, class FB >
KeopsNS<Scalprod<FA,FB>> operator|(KeopsNS<FA> fa, KeopsNS<FB> fb) {
    return KeopsNS<Scalprod<FA,FB>>();
}

// reductions

#define SumReduction(F,I) KeopsNS<SumReduction<decltype(InvKeopsNS(F)),I>>()
#define LogSumExpReduction(F,I) KeopsNS<LogSumExpReduction<decltype(InvKeopsNS(F)),I>>()

#define MinArgMinReduction(F,I) KeopsNS<MinArgMinReduction<decltype(InvKeopsNS(F)),I>>()
#define ArgMinReduction(F,I) KeopsNS<ArgMinReduction<decltype(InvKeopsNS(F)),I>>()
#define MinReduction(F,I) KeopsNS<MinReduction<decltype(InvKeopsNS(F)),I>>()

#define MaxArgMaxReduction(F,I) KeopsNS<MaxArgMaxReduction<decltype(InvKeopsNS(F)),I>>()
#define ArgMaxReduction(F,I) KeopsNS<ArgMaxReduction<decltype(InvKeopsNS(F)),I>>()
#define MaxReduction(F,I) KeopsNS<MaxReduction<decltype(InvKeopsNS(F)),I>>()

#define KMinArgKMinReduction(F,K,I) KeopsNS<KMinArgKMinReduction<decltype(InvKeopsNS(F)),K,I>>()
#define ArgKMinReduction(F,K,I) KeopsNS<ArgKMinReduction<decltype(InvKeopsNS(F)),K,I>>()
#define KMinReduction(F,K,I) KeopsNS<KMinReduction<decltype(InvKeopsNS(F)),K,I>>()

}
