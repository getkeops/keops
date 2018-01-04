#ifndef NEWSYNTAX
#define NEWSYNTAX

#define x0(DIM) _X<0,DIM>()
#define x1(DIM) _X<1,DIM>()
#define x2(DIM) _X<2,DIM>()
#define x3(DIM) _X<3,DIM>()
#define x4(DIM) _X<4,DIM>()
#define x5(DIM) _X<5,DIM>()
#define x6(DIM) _X<6,DIM>()
#define x7(DIM) _X<7,DIM>()
#define x8(DIM) _X<8,DIM>()
#define x9(DIM) _X<9,DIM>()

#define y0(DIM) _Y<0,DIM>()
#define y1(DIM) _Y<1,DIM>()
#define y2(DIM) _Y<2,DIM>()
#define y3(DIM) _Y<3,DIM>()
#define y4(DIM) _Y<4,DIM>()
#define y5(DIM) _Y<5,DIM>()
#define y6(DIM) _Y<6,DIM>()
#define y7(DIM) _Y<7,DIM>()
#define y8(DIM) _Y<8,DIM>()
#define y9(DIM) _Y<9,DIM>()

#define p0 _P<0>()
#define p1 _P<1>()
#define p2 _P<2>()
#define p3 _P<3>()
#define p4 _P<4>()
#define p5 _P<5>()
#define p6 _P<6>()
#define p7 _P<7>()
#define p8 _P<8>()
#define p9 _P<9>()


#define Factorize(F,G) Factorize<decltype(F),decltype(G)>()

#define Grad(F,V,GRADIN)  Grad<decltype(F),decltype(V),decltype(GRADIN)>()

#define IntCst(N) IntConstant<N>
#define Cst(p) Constant<decltype(p)>()

template < class FA, class FB >
Add<FA,FB> operator+(FA fa, FB fb)
{
	return Add<FA,FB>();
}

template < class FA, class FB >
Scal<FA,FB> operator*(FA fa, FB fb)
{
	return Scal<FA,FB>();
}

#define Exp(f) Exp<decltype(f)>()

#define Pow(f,M) Pow<decltype(f),M>()

#define Square(f) Square<decltype(f)>()

template < class F >
Minus<F> operator-(F f)
{
	return Minus<F>();
}

template < class FA, class FB >
Subtract<FA,FB> operator-(FA fa, FB fb)
{
	return Subtract<FA,FB>();
}

#define Inv(f) Inv<decltype(f)>()

#define IntInv(f) IntInv<decltype(f)>()

template < class FA, class FB >
Divide<FA,FB> operator/(FA fa, FB fb)
{
	return Divide<FA,FB>();
}

#define Log(f) Log<decltype(f)>()

#define Powf(fa,fb) Powf<decltype(fa),decltype(fb)>()

#define Sqrt(f) Sqrt<decltype(f)>()

template < class FA, class FB >
Scalprod<FA,FB> operator,(FA fa, FB fb)
{
	return Scalprod<FA,FB>();
}

#define SqNorm2(f) SqNorm2<decltype(f)>()

#define SqDist(f,g) SqDist<decltype(f),decltype(g)>()



#define GaussKernel(OOS2,X,Y,Beta) GaussKernel<decltype(OOS2),decltype(X),decltype(Y),decltype(Beta)>()

#define GaussKernel_(DIMPOINT,DIMVECT) GaussKernel_<DIMPOINT,DIMVECT>()
#define LaplaceKernel(DIMPOINT,DIMVECT) LaplaceKernel<DIMPOINT,DIMVECT>()
#define EnergyKernel(DIMPOINT,DIMVECT) EnergyKernel<DIMPOINT,DIMVECT>()

#endif // NEWSYNTAX