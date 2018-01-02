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
////                 FACTORIZED GAUSS KERNEL              ////
//////////////////////////////////////////////////////////////
template < int DIMPOINT, int DIMVECT >
using GaussKernel_Factorized = Factorize< GaussKernel_<DIMPOINT,DIMVECT> , Subtract<_X<0,DIMPOINT>,_Y<1,DIMPOINT>> >;


//////////////////////////////////////////////////////////////
////   DIRECT IMPLEMENTATIONS FOR SCALAR RADIAL KERNELS   ////
////	(FOR FASTER COMPUTATIONS)                         ////
//////////////////////////////////////////////////////////////

// specific implementation of the gaussian kernel and its gradient wrt to X

template < int DIMPOINT, int DIMVECT > struct GaussKernel_specific;
template < int DIMPOINT, int DIMVECT, class V, class GRADIN > struct GradGaussKernel_specific;

template < int DIMPOINT, int DIMVECT >
struct GaussKernel_specific {
    using GenericVersion = GaussKernel_<DIMPOINT,DIMVECT>;

    static const int DIM = GenericVersion::DIM;
	
	template < int CAT >
	using VARS = typename GenericVersion::template VARS<CAT>;

    using THIS = GaussKernel_specific<DIMPOINT,DIMVECT>;
		
    template<class A, class B>
    using Replace = CondType< B, THIS, IsSameType<A,THIS>::val >;
    
    using AllTypes = univpack<THIS>;

    template < class INDS, typename... ARGS >
    INLINE void Eval(__TYPE__* params, __TYPE__* gammai, ARGS... args) {
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

// by default we link to the standard autodiff versions of the gradients
template < int DIMPOINT, int DIMVECT, class V, class GRADIN >
struct GradGaussKernel_specific {
    using GenericVersion = Grad<GaussKernel_<DIMPOINT,DIMVECT>,V,GRADIN>;

    static const int DIM = GenericVersion::DIM;
	
	template < int CAT >
	using VARS = typename GenericVersion::template VARS<CAT>;
		
    using THIS = GradGaussKernel_specific<DIMPOINT,DIMVECT,V,GRADIN>;
		
    template<class A, class B>
    using Replace = CondType< B, GradGaussKernel_specific<DIMPOINT,DIMVECT,V,typename GRADIN::Replace<A,B>>, IsSameType<A,THIS>::val >;
    
    using AllTypes = MergePacks < univpack<THIS,V> , typename GRADIN::AllTypes >;

    template < class INDS, typename... ARGS >
    INLINE void Eval(__TYPE__* params, __TYPE__* gammai, ARGS... args) {
        GenericVersion::template Eval<INDS>(params,gammai,args...);
    }

    template < class V2, class GRADIN2 >
    using DiffT = Grad<GenericVersion,V2,GRADIN2>;

};

// specific implementation of gradient wrt X
template < int DIMPOINT, int DIMVECT, class GRADIN >
struct GradGaussKernel_specific<DIMPOINT,DIMVECT,_X<0,DIMPOINT>,GRADIN> {
    using GenericVersion = Grad<GaussKernel_<DIMPOINT,DIMVECT>,_X<0,DIMPOINT>,GRADIN>;

    static const int DIM = GenericVersion::DIM;
	
	template < int CAT >
	using VARS = typename GenericVersion::template VARS<CAT>;
		
    using THIS = GradGaussKernel_specific<DIMPOINT,DIMVECT,_X<0,DIMPOINT>,GRADIN>;
		
    template<class A, class B>
    using Replace = CondType< B, GradGaussKernel_specific<DIMPOINT,DIMVECT,_X<0,DIMPOINT>,typename GRADIN::Replace<A,B>>, IsSameType<A,THIS>::val >;
    
    using AllTypes = MergePacks < univpack<THIS,_X<0,DIMPOINT>> , typename GRADIN::AllTypes >;

    template < class INDS, typename... ARGS >
    INLINE void Eval(__TYPE__* params, __TYPE__* gammai, ARGS... args) {
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
    using DiffT = Grad<GenericVersion,V2,GRADIN2>;

};



//////////////////////////////////////////////////////////////
////                 MATRIX-VALUED KERNELS                ////
//////////////////////////////////////////////////////////////

// Matrix-valued kernels : implementations from Micheli/Glaunes paper

// TRI kernel - general form :
// k(x,y)b = k_ortho(r2)b + k_tilde(r2)<b,x-y>(x-y),   where r2=|x-y|^2
// which gives the formula below. 

// we construct the formula step by step (we use a struct just as a namespace, to avoid defining these temporary alias in the global scope)
template < template<class,class> class FORTHO, template<class,class> class FTILDE, int DIM >
struct TRI_Kernel_helper
{
	using R2 = SqDist<_X<0,DIM>,_Y<1,DIM>>; 	// r2=|x-y|^2
	using KORTHOR2 = FORTHO<Param<0>,R2>;		// k_ortho(r2)
	using B = _X<2,DIM>;						// b
	using KTILDER2 = FTILDE<Param<1>,R2>;		// k_tilde(r2)
	using XMY = Subtract<_X<0,DIM>,_Y<1,DIM>>;	// x-y
	using BDOTXMY = Scalprod<_X<2,DIM>,XMY>;	// <b,x-y>
	using C = Scalprod<BDOTXMY,XMY>;			// <b,x-y>(x-y)
	using type = Add<Scal<KORTHOR2,B>,Scal<KTILDER2,C>>;		// final formula 
	using factorized_type = Factorize<Factorize<type,R2>,XMY>;	// formula, factorized by r2 and x-y
};

// final definition is here
template < template<class,class> class FORTHO, template<class,class> class FTILDE, int DIM >
using TRI_Kernel = typename TRI_Kernel_helper<FORTHO,FTILDE,DIM>::factorized_type;

// Div-free and curl-free kernel with gaussian functions.
// k_df(x,y)b = exp(-r^2/s2)*(((d-1)/(2c)-r^2)b + <b,x-y>(x-y))
// k_cf(x,y)b = exp(-r^2/s2)*(	   (1/(2c)) b   - <b,x-y>(x-y))
// The value of 1/s2 must be given as first parameter (_P<0>) when calling Eval()
// We do not use the previous template because exp(-r^2/s2) is factorized

template < int DIM >
struct DivFreeGaussKernel_helper
{
	using R2 = SqDist<_X<0,DIM>,_Y<1,DIM>>; 			// r2=|x-y|^2
	using XMY = Subtract<_X<0,DIM>,_Y<1,DIM>>;			// x-y
	using G = GaussFunction<_P<0>,R2>;					// exp(-r^2/s2)
	using TWOC = Scal<IntConstant<2>,Constant<_P<0>>>; 	// 2c
	using C1 = Divide<IntConstant<DIM-1>,TWOC>;			// (d-1)/(2c)
	using B = _X<2,DIM>;								// b
	using C2 = Scal<Subtract<C1,R2>,B>;					// ((d-1)/(2c)-r^2)b
	using BDOTXMY = Scalprod<_X<2,DIM>,XMY>;			// <b,x-y>
	using C = Scal<BDOTXMY,XMY>;						// <b,x-y>(x-y)
	using type = Scal<G,Add<C2,C>>;								// final formula
	using factorized_type = Factorize<Factorize<type,R2>,XMY>;	// formula, factorized by r2 and x-y
};

template < int DIM >
using DivFreeGaussKernel = typename DivFreeGaussKernel_helper<DIM>::factorized_type;

template < int DIM >
struct CurlFreeGaussKernel_helper
{
	using R2 = SqDist<_X<0,DIM>,_Y<1,DIM>>; 			// r2=|x-y|^2
	using XMY = Subtract<_X<0,DIM>,_Y<1,DIM>>;			// x-y
	using G = GaussFunction<_P<0>,R2>;					// exp(-r^2/s2)
	using TWOC = Scal<IntConstant<2>,Constant<_P<0>>>; 	// 2c
	using C1 = Divide<IntConstant<1>,TWOC>;				// 1/(2c)
	using B = _X<2,DIM>;								// b
	using C2 = Scal<C1,B>;								// (1/(2c))b
	using BDOTXMY = Scalprod<_X<2,DIM>,XMY>;			// <b,x-y>
	using C = Scal<BDOTXMY,XMY>;						// <b,x-y>(x-y)
	using type = Scal<G,Subtract<C2,C>>;						// final formula
	using factorized_type = Factorize<Factorize<type,R2>,XMY>;	// formula, factorized by r2 and x-y
};

template < int DIM >
using CurlFreeGaussKernel = typename CurlFreeGaussKernel_helper<DIM>::factorized_type;

// Weighted combination of the two previous kernels, which gives a Translation and Rotation Invariant kernel with gaussian base function.
// k_tri(x,y)b = lambda * k_df(x,y)b + (1-lambda) * k_cf(x,y)b
// The weight lambda must be specified as the second parameter (_P<1>) when calling Eval()

template < int DIM >
struct TRIGaussKernel_helper
{
	using L = Constant<_P<1>>;								// lambda
	using OML = Subtract<IntConstant<1>,L>;					// 1-lambda
	using DF = DivFreeGaussKernel_helper<DIM>;				// k_df(x,y)b (the helper struct, because we need it below)
	using CF = CurlFreeGaussKernel_helper<DIM>;				// k_cf(x,y)b (the helper struct, because we need it below)
	using type =  Add<Scal<L,typename DF::type>,Scal<OML,typename CF::type>>;	// final formula (not factorized)
	// here we can factorize a lot ; we look at common expressions in Div Free and Curl Free kernels:
	using G = typename DF::G;							// exp(-r^2/s2) can be factorized
	using C = typename DF::C;							// <b,x-y>(x-y) can be factorized
	using XMY = typename DF::XMY;						// x-y can be factorized
	using R2 = typename DF::R2;							// r^2 can be factorized
	using factorized_type = Factorize<Factorize<Factorize<Factorize<type,G>,C>,R2>,XMY>;
};
	
template < int DIM >
using TRIGaussKernel = typename TRIGaussKernel_helper<DIM>::factorized_type;
