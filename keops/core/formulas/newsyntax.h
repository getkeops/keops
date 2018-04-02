#pragma once

#include "core/formulas/constants.h"
#include "core/formulas/maths.h"
#include "core/formulas/kernels.h"
#include "core/formulas/norms.h"
#include "core/formulas/factorize.h"

#include "core/reductions/sum.h"
#include "core/reductions/log_sum_exp.h"



#define Var(N,DIM, CAT) Var<N,DIM,CAT>()

#define Vx(N,DIM) _X<N,DIM>()

#define Vy(N,DIM) _Y<N,DIM>()

#define Pm(N,DIM) _P<N,DIM>()

#define LogSumExp(f) LogSumExp<decltype(f)>()

#define Factorize(F,G) Factorize<decltype(F),decltype(G)>()

#define AutoFactorize(F) AutoFactorize<decltype(F)>()

#define Grad(F,V,GRADIN)  Grad<decltype(F),decltype(V),decltype(GRADIN)>()

#define IntCst(N) IntConstant<N>()

#define Elem(p,k) Elem<decltype(p),k>()

#define ElemT(p,k) ElemT<decltype(p),k>()

template < class FA, class FB >
Add<FA,FB> operator+(FA fa, FB fb) {
    return Add<FA,FB>();
}

template < class FA, class FB >
ScalOrMult<FA,FB> operator*(FA fa, FB fb) {
    return ScalOrMult<FA,FB>();
}

#define Exp(f) Exp<decltype(f)>()

#define Pow(f,M) Pow<decltype(f),M>()

#define Square(f) Square<decltype(f)>()

template < class F >
Minus<F> operator-(F f) {
    return Minus<F>();
}

template < class FA, class FB >
Subtract<FA,FB> operator-(FA fa, FB fb) {
    return Subtract<FA,FB>();
}

#define Inv(f) Inv<decltype(f)>()

#define IntInv(N) IntInv<N>()

template < class FA, class FB >
Divide<FA,FB> operator/(FA fa, FB fb) {
    return Divide<FA,FB>();
}

#define Log(f) Log<decltype(f)>()

#define Powf(fa,fb) Powf<decltype(fa),decltype(fb)>()

#define Sqrt(f) Sqrt<decltype(f)>()

template < class FA, class FB >
Scalprod<FA,FB> operator,(FA fa, FB fb) {
    return Scalprod<FA,FB>();
}

#define SqNorm2(f) SqNorm2<decltype(f)>()
#define SqDist(f,g) SqDist<decltype(f),decltype(g)>()

#define WeightedSqNorm(s,f)   WeightedSqNorm<decltype(s), decltype(f)>()
#define WeightedSqDist(s,f,g) WeightedSqDist<decltype(s), decltype(f),decltype(g)>()



#define GaussKernel(C,X,Y,B) GaussKernel<decltype(C),decltype(X),decltype(Y),decltype(B)>()
#define CauchyKernel(C,X,Y,B) CauchyKernel<decltype(C),decltype(X),decltype(Y),decltype(B)>()
#define LaplaceKernel(C,X,Y,B) LaplaceKernel<decltype(C),decltype(X),decltype(Y),decltype(B)>()
#define InverseMultiquadricKernel(C,X,Y,B) InverseMultiquadricKernel<decltype(C),decltype(X),decltype(Y),decltype(B)>()
#define SumGaussKernel(C,W,X,Y,B) SumGaussKernel<decltype(C),decltype(W),decltype(X),decltype(Y),decltype(B)>()

#define DivFreeGaussKernel(C,X,Y,B) DivFreeGaussKernel<decltype(C),decltype(X),decltype(Y),decltype(B)>()
#define CurlFreeGaussKernel(C,X,Y,B) CurlFreeGaussKernel<decltype(C),decltype(X),decltype(Y),decltype(B)>()
#define TRIGaussKernel(L,C,X,Y,B) TRIGaussKernel<decltype(L),decltype(C),decltype(X),decltype(Y),decltype(B)>()

