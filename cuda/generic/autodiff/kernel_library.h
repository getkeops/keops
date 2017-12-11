// Aliases for custom kernels


// Scalar radial kernels

template < template<class,class> class F, int DIMPOINT, int DIMVECT >
using ScalarRadialKernel = Scal<F<Param<0>,SqDist<X<0,DIMPOINT>,Y<1,DIMPOINT>>>,Y<2,DIMVECT>>;

template < class OOS2, class R2 >
using GaussFunction = Exp<Scal<Constant<OOS2>,Minus<R2>>>;

template < class OOS2, class R2 >
using LaplaceFunction = Exp<Minus<Sqrt<Add<Inv<Constant<OOS2>>,R2>>>>;

template < class OOS2, class R2 >
using EnergyFunction = Inv<Powf<Add<Inv<Constant<OOS2>>,R2>,IntInv<4>>>;

template < int DIMPOINT, int DIMVECT >
using GaussKernel_ = ScalarRadialKernel<GaussFunction,DIMPOINT,DIMVECT>;
// remark : GaussKernel is already defined with different template arguments ; this one should replace the first

template < int DIMPOINT, int DIMVECT >
using LaplaceKernel = ScalarRadialKernel<LaplaceFunction,DIMPOINT,DIMVECT>;

template < int DIMPOINT, int DIMVECT >
using EnergyKernel = ScalarRadialKernel<EnergyFunction,DIMPOINT,DIMVECT>;


// Matrix-valued kernels : implementations from Micheli/Glaunes paper

// TRI kernel - general form :
// k(x,y)b = k_ortho(r2)b + k_tilde(r2)<b,x-y>(x-y),   where r2=|x-y|^2
// which gives the formula below. Drawback : x-y and r2 are computed several times when calling Eval()
// Is there a way to avoid this by detecting several Eval() calls for the same class ??
template < template<class,class> class FORTHO, template<class,class> class FTILDE, int DIM >
using TRI_Kernel = Add<Scal<FORTHO<Param<0>,SqDist<X<0,DIM>,Y<1,DIM>>>,X<2,DIM>>,
	Scal<Scal<FTILDE<Param<1>,SqDist<X<0,DIM>,Y<1,DIM>>>,Scalprod
			<X<2,DIM>,Subtract<X<0,DIM>,Y<1,DIM>>>>,
		Subtract<X<0,DIM>,Y<1,DIM>>>>;


// Div-free and curl-free kernel with gaussian functions. 
// k_df(x,y)b = exp(-r^2/s2)*(((d-1)/(2c)-r^2)b + <b,x-y>(x-y))
// k_cf(x,y)b = exp(-r^2/s2)*(	   (1/(2c)) b   - <b,x-y>(x-y))
// We do not use the previous template because exp(-r^2/s2) is factorized

template < int DIM >
using DivFreeGaussKernel = 
Scal<GaussFunction<P<0>,SqDist<X<0,DIM>,Y<1,DIM>>>,
Add<Scal<Subtract<
Divide<IntConstant<DIM-1>,Scal<IntConstant<2>,Constant<P<0>>>>,
SqDist<X<0,DIM>,Y<1,DIM>>>,Y<2,DIM>>,
Scal<Scalprod<Y<2,DIM>,Subtract<X<0,DIM>,Y<1,DIM>>>,Subtract<X<0,DIM>,Y<1,DIM>>>>>;

template < int DIM >
using CurlFreeGaussKernel = 
Scal<GaussFunction<P<0>,SqDist<X<0,DIM>,Y<1,DIM>>>,
Subtract<Scal<
Divide<IntConstant<1>,Scal<IntConstant<2>,Constant<P<0>>>>,
Y<2,DIM>>,
Scal<Scalprod<Y<2,DIM>,Subtract<X<0,DIM>,Y<1,DIM>>>,Subtract<X<0,DIM>,Y<1,DIM>>>>>;

