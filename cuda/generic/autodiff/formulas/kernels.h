/*
 * The file where the most useful kernel-related operators are defined.
 * 
 * Available kernel-related routines are :
 *   Radial functions :
 *      GaussFunction<C,R2>						: = exp( - C * R2 )
 *      LaplaceFunction<C,R2>					: = exp( - sqrt( 1/C + R2 ) )
 *      EnergyFunction<C,R2>					: = (1/C + R2)^(-1/4)
 * 
 *   Utility functions :
 *      ScalarRadialKernel<F,DIMPOINT,DIMVECT>	: which builds a function 
 *                                                (x_i,y_j,b_j) -> F_s( |x_i-y_j|^2 ) * b_j from
 *                                                a radial function F<S,R2> -> ...,
 *                                                a "point" dimension DIMPOINT (x_i and y_j)
 *                                                a "vector" dimension DIMVECT (b_j and output)
 *   
 *   Radial Kernel operators : inline expressions w.r.t. x_i = X_0, y_j = Y_1, b_j = Y_2
 *      GaussKernel<DIMPOINT,DIMVECT>			: uses GaussFunction
 *      LaplaceKernel<DIMPOINT,DIMVECT>			: uses LaplaceFunction
 *      EnergyKernel<DIMPOINT,DIMVECT>			: uses EnergyFunction
 * 
 */

//////////////////////////////////////////////////////////////
////             STANDARD RADIAL FUNCTIONS                ////
//////////////////////////////////////////////////////////////

template < class OOS2, class R2 >
using GaussFunction = Exp<Scal<Constant<OOS2>,Minus<R2>>>;

template < class OOS2, class R2 >
using LaplaceFunction = Exp<Minus<Sqrt<Add<Inv<Constant<OOS2>>,R2>>>>;

template < class OOS2, class R2 >
using EnergyFunction = Inv<Powf<Add<Inv<Constant<OOS2>>,R2>,IntInv<4>>>;


//////////////////////////////////////////////////////////////
////                 SCALAR RADIAL KERNELS                ////
//////////////////////////////////////////////////////////////

// Utility function
template < template<class,class> class F, int DIMPOINT, int DIMVECT >
using ScalarRadialKernel = Scal<F<Param<0>,SqDist<X<0,DIMPOINT>,Y<1,DIMPOINT>>>,Y<2,DIMVECT>>;

// Utility aliases :
template < int DIMPOINT, int DIMVECT >
using GaussKernel = ScalarRadialKernel<GaussFunction,DIMPOINT,DIMVECT>;

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
// The value of 1/s2 must be given as first parameter (P<0>) when calling Eval() 
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

// Weighted combination of the two previous kernels, which gives a Translation and Rotation Invariant kernel with gaussian base function.
// k_tri(x,y)b = lambda * k_df(x,y)b + (1-lambda) * k_cf(x,y)b
// The weight lambda must be specified as the second parameter (P<1>) when calling Eval() 
// remark : this is currently not efficient at all since almost the same computations will be done twice...

template < int DIM >
using TRIGaussKernel = Add<Scal<Constant<P<1>>,DivFreeGaussKernel<DIM>>,Scal<Subtract<IntConstant<1>,Constant<P<1>>>,CurlFreeGaussKernel<DIM>>>;
