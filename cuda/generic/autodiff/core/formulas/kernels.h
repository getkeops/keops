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

// Legacy 

template < class OOS2, class X, class Y, class Beta > 
using GaussKernel = Scal<Exp<Scal<Constant<OOS2>,Minus<SqDist<X,Y>>>>,Beta>;

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
using ScalarRadialKernel = Scal<F<Param<0>,SqDist<_X<0,DIMPOINT>,_Y<1,DIMPOINT>>>,_Y<2,DIMVECT>>;

// Utility aliases :
template < int DIMPOINT, int DIMVECT >
using GaussKernel_ = ScalarRadialKernel<GaussFunction,DIMPOINT,DIMVECT>;

template < int DIMPOINT, int DIMVECT >
using LaplaceKernel = ScalarRadialKernel<LaplaceFunction,DIMPOINT,DIMVECT>;

template < int DIMPOINT, int DIMVECT >
using EnergyKernel = ScalarRadialKernel<EnergyFunction,DIMPOINT,DIMVECT>;


//////////////////////////////////////////////////////////////
////   DIRECT IMPLEMENTATIONS FOR SCALAR RADIAL KERNELS   ////
//////////////////////////////////////////////////////////////

// hardcoded implementation of the gaussian kernel

template < int DIMPOINT, int DIMVECT > struct GaussKernel_specific;
template < int DIMPOINT, int DIMVECT, class V, class GRADIN > struct GradGaussKernel_specific;

template < int DIMPOINT, int DIMVECT >
struct GaussKernel_specific
{
    static const int DIM = DIMVECT;
	
	using VARS = univpack < X<0,DIMPOINT> , Y<1,DIMPOINT> , Y<2,DIMVECT> >;
	
    template < class INDS, typename... ARGS >
    INLINE void Eval(__TYPE__* params, __TYPE__* gammai, ARGS... args)
	{	
        auto t = thrust::make_tuple(args...); 
        __TYPE__*& xi = thrust::get<IndValAlias<INDS,0>::ind>(t); 
        __TYPE__*& yj = thrust::get<IndValAlias<INDS,1>::ind>(t); 
        __TYPE__*& betaj = thrust::get<IndValAlias<INDS,2>::ind>(t); 

	    __TYPE__ r2 = 0.0f;
        __TYPE__ temp;
        for(int k=0; k<DIMPOINT; k++) {
            temp =  yj[k]-xi[k];
            r2 += temp*temp;
        }
        __TYPE__ s = exp(-r2*params[0]);
        for(int k=0; k<DIMVECT; k++)
            gammai[k] += s * betaj[k];
    }

    template < class V, class GRADIN >
    using DiffT = GradGaussKernel_specific<DIMPOINT,DIMVECT,V,GRADIN>;

};	

// not all gradients are implemented yet, so by default we make links to the standard autodiff versions
template < int DIMPOINT, int DIMVECT, class V, class GRADIN > 
struct GradGaussKernel_specific
{
    static const int DIM = DIMPOINT;
	
	using VARS = MergePacks < typename GaussKernel_specific<DIMPOINT,DIMVECT>::VARS , typename GRADIN::VARS >;
	
    template < class INDS, typename... ARGS >
    INLINE void Eval(__TYPE__* params, __TYPE__* gammai, ARGS... args)
	{	
        Grad<GaussKernel_<DIMPOINT,DIMVECT>,V,GRADIN>::template Eval<INDS>(params,gammai,args...);
    }

    template < class V2, class GRADIN2 >
    using DiffT = Grad<Grad<GaussKernel_<DIMPOINT,DIMVECT>,V,GRADIN>,V2,GRADIN2>;
	
};
	


// implementation of gradient wrt X

template < int DIMPOINT, int DIMVECT, class GRADIN > 
struct GradGaussKernel_specific<DIMPOINT,DIMVECT,X<0,DIMPOINT>,GRADIN>
{
    static const int DIM = DIMPOINT;
	
	using VARS = MergePacks < typename GaussKernel_specific<DIMPOINT,DIMVECT>::VARS , typename GRADIN::VARS >;
		
    template < class INDS, typename... ARGS >
    INLINE void Eval(__TYPE__* params, __TYPE__* gammai, ARGS... args)
	{	
        auto t = thrust::make_tuple(args...); 
        __TYPE__*& xi = thrust::get<IndValAlias<INDS,0>::ind>(t); 
        __TYPE__*& yj = thrust::get<IndValAlias<INDS,1>::ind>(t); 
        __TYPE__*& betaj = thrust::get<IndValAlias<INDS,2>::ind>(t); 
        __TYPE__ alphai[GRADIN::DIM];
        GRADIN::template Eval<INDS>(params,alphai,args...);

        __TYPE__ r2 = 0.0f, sga = 0.0f;          // Don't forget to initialize at 0.0
        __TYPE__ xmy[DIMPOINT];
        for(int k=0; k<DIMPOINT; k++) {      // Compute the L2 squared distance r2 = | x_i-y_j |_2^2
            xmy[k] =  xi[k]-yj[k];
            r2 += xmy[k]*xmy[k];
        }
        for(int k=0; k<DIMVECT; k++)         // Compute the L2 dot product <a_i, b_j>
            sga += betaj[k]*alphai[k];
        __TYPE__ s = - 2.0 * sga * exp(-r2*params[0]);  // Don't forget the 2 !
        for(int k=0; k<DIMPOINT; k++)        // Increment the output vector gammai - which is a POINT
            gammai[k] += s * xmy[k];
    }

	// direct implementation stops here, so we link back to the usual autodiff module
    template < class V2, class GRADIN2 >
    using DiffT = Grad<Grad<GaussKernel_<DIMPOINT,DIMVECT>,X<0,DIMPOINT>,GRADIN>,V2,GRADIN2>;
	
};
	
	

//////////////////////////////////////////////////////////////
////                 MATRIX-VALUED KERNELS                ////
//////////////////////////////////////////////////////////////

// Matrix-valued kernels : implementations from Micheli/Glaunes paper

// TRI kernel - general form :
// k(x,y)b = k_ortho(r2)b + k_tilde(r2)<b,x-y>(x-y),   where r2=|x-y|^2
// which gives the formula below. Drawback : x-y and r2 are computed several times when calling Eval()
// Is there a way to avoid this by detecting several Eval() calls for the same class ??
template < template<class,class> class FORTHO, template<class,class> class FTILDE, int DIM >
using TRI_Kernel = Add<Scal<FORTHO<Param<0>,SqDist<_X<0,DIM>,_Y<1,DIM>>>,_X<2,DIM>>,
	Scal<Scal<FTILDE<Param<1>,SqDist<_X<0,DIM>,_Y<1,DIM>>>,Scalprod
			<_X<2,DIM>,Subtract<_X<0,DIM>,_Y<1,DIM>>>>,
		Subtract<_X<0,DIM>,_Y<1,DIM>>>>;


// Div-free and curl-free kernel with gaussian functions. 
// k_df(x,y)b = exp(-r^2/s2)*(((d-1)/(2c)-r^2)b + <b,x-y>(x-y))
// k_cf(x,y)b = exp(-r^2/s2)*(	   (1/(2c)) b   - <b,x-y>(x-y))
// The value of 1/s2 must be given as first parameter (P<0>) when calling Eval() 
// We do not use the previous template because exp(-r^2/s2) is factorized

template < int DIM >
using DivFreeGaussKernel = 
Scal<GaussFunction<_P<0>,SqDist<_X<0,DIM>,_Y<1,DIM>>>,
Add<Scal<Subtract<
Divide<IntConstant<DIM-1>,Scal<IntConstant<2>,Constant<_P<0>>>>,
SqDist<_X<0,DIM>,_Y<1,DIM>>>,_Y<2,DIM>>,
Scal<Scalprod<_Y<2,DIM>,Subtract<_X<0,DIM>,_Y<1,DIM>>>,Subtract<_X<0,DIM>,_Y<1,DIM>>>>>;

template < int DIM >
using CurlFreeGaussKernel = 
Scal<GaussFunction<_P<0>,SqDist<_X<0,DIM>,_Y<1,DIM>>>,
Subtract<Scal<
Divide<IntConstant<1>,Scal<IntConstant<2>,Constant<_P<0>>>>,
_Y<2,DIM>>,
Scal<Scalprod<_Y<2,DIM>,Subtract<_X<0,DIM>,_Y<1,DIM>>>,Subtract<_X<0,DIM>,_Y<1,DIM>>>>>;

// Weighted combination of the two previous kernels, which gives a Translation and Rotation Invariant kernel with gaussian base function.
// k_tri(x,y)b = lambda * k_df(x,y)b + (1-lambda) * k_cf(x,y)b
// The weight lambda must be specified as the second parameter (P<1>) when calling Eval() 
// remark : this is currently not efficient at all since almost the same computations will be done twice...

template < int DIM >
using TRIGaussKernel = Add<Scal<Constant<_P<1>>,DivFreeGaussKernel<DIM>>,Scal<Subtract<IntConstant<1>,Constant<_P<1>>>,CurlFreeGaussKernel<DIM>>>;
