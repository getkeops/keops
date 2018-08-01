#pragma once

#include "core/formulas/constants.h"
#include "core/formulas/maths.h"
#include "core/formulas/kernels.h"
#include "core/formulas/norms.h"
#include "core/formulas/factorize.h"

#include "core/reductions/sum.h"
#include "core/reductions/log_sum_exp.h"

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

#define Grad(F,V,GRADIN)  KeopsNS<Grad<decltype(InvKeopsNS(F)),InvKeopsNS(V),decltype(InvKeopsNS(GRADIN))>>()

#define IntCst(N) KeopsNS<IntConstant<N>>()

#define Elem(p,k) KeopsNS<Elem<decltype(InvKeopsNS(p)),k>>()

#define ElemT(p,k) KeopsNS<ElemT<decltype(InvKeopsNS(p)),k>>()

#define Exp(f) KeopsNS<Exp<decltype(InvKeopsNS(f))>>()

#define Pow(f,M) KeopsNS<Pow<decltype(InvKeopsNS(f)),M>>()

#define Square(f) KeopsNS<Square<decltype(InvKeopsNS(f))>>()

#define Inv(f) KeopsNS<Inv<decltype(InvKeopsNS(f))>>()

#define IntInv(N) KeopsNS<IntInv<N>>()

#define Log(f) KeopsNS<Log<decltype(InvKeopsNS(f))>>()

#define Powf(fa,fb) KeopsNS<Powf<decltype(InvKeopsNS(fa)),decltype(InvKeopsNS(fb))>>()

#define Sqrt(f) KeopsNS<Sqrt<decltype(InvKeopsNS(f))>>()

#define SqNorm2(f) KeopsNS<SqNorm2<decltype(InvKeopsNS(f))>>()
#define SqDist(f,g) KeopsNS<SqDist<decltype(InvKeopsNS(f)),decltype(InvKeopsNS(g))>>()

#define WeightedSqNorm(s,f)   KeopsNS<WeightedSqNorm<decltype(InvKeopsNS(s)), decltype(InvKeopsNS(f))>>()
#define WeightedSqDist(s,f,g) KeopsNS<WeightedSqDist<decltype(InvKeopsNS(s)), decltype(InvKeopsNS(f)),decltype(InvKeopsNS(g))>>()



#define GaussKernel(C,X,Y,B) KeopsNS<GaussKernel<InvKeopsNS(C),InvKeopsNS(X),InvKeopsNS(Y),InvKeopsNS(B)>>()
#define CauchyKernel(C,X,Y,B) KeopsNS<CauchyKernel<InvKeopsNS(C),InvKeopsNS(X),InvKeopsNS(Y),InvKeopsNS(B)>>()
#define LaplaceKernel(C,X,Y,B) KeopsNS<LaplaceKernel<InvKeopsNS(C),InvKeopsNS(X),InvKeopsNS(Y),InvKeopsNS(B)>>()
#define InverseMultiquadricKernel(C,X,Y,B) KeopsNS<InverseMultiquadricKernel<InvKeopsNS(C),InvKeopsNS(X),InvKeopsNS(Y),InvKeopsNS(B)>>()
#define SumGaussKernel(C,W,X,Y,B) KeopsNS<SumGaussKernel<InvKeopsNS(C),InvKeopsNS(W),InvKeopsNS(X),InvKeopsNS(Y),InvKeopsNS(B)>>()

#define DivFreeGaussKernel(C,X,Y,B) KeopsNS<DivFreeGaussKernel<InvKeopsNS(C),InvKeopsNS(X),InvKeopsNS(Y),InvKeopsNS(B)>>()
#define CurlFreeGaussKernel(C,X,Y,B) KeopsNS<CurlFreeGaussKernel<InvKeopsNS(C),InvKeopsNS(X),InvKeopsNS(Y),InvKeopsNS(B)>>()
#define TRIGaussKernel(L,C,X,Y,B) KeopsNS<TRIGaussKernel<InvKeopsNS(L),InvKeopsNS(C),InvKeopsNS(X),InvKeopsNS(Y),InvKeopsNS(B)>>()


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

}
