#pragma once

#include <sstream>

#include "core/Pack.h"
#include "core/autodiff.h"

#include "core/formulas/constants.h"
#include "core/formulas/maths.h"
#include "core/formulas/norms.h"
#include "core/formulas/factorize.h"

/*
 * The file where the most useful kernel-related operators are defined.
 *
 * Available kernel-related routines are :
 *   Radial functions :
 *      GaussFunction<R2,C>						: = exp( - C * R2 )
 *      CauchyFunction<R2,C>					: = 1/( 1 +  R2 * C )
 *      LaplaceFunction<R2,C>					: = exp( - sqrt( C * R2 ) )
 *      InverseMultiquadricFunction<R2,C>				: = (1/C + R2)^(-1/2)
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
 *      CauchyKernel<DIMPOINT,DIMVECT>			: uses CauchyFunction
 *      LaplaceKernel<DIMPOINT,DIMVECT>			: uses LaplaceFunction
 *      InverseMultiquadricKernel<DIMPOINT,DIMVECT>	: uses InverseMultiquadricFunction
 *
 */

//////////////////////////////////////////////////////////////
////             STANDARD RADIAL FUNCTIONS                ////
//////////////////////////////////////////////////////////////
namespace keops {
template < class R2, class C >
using GaussFunction = Exp<Scal<C,Minus<R2>>>;

template < class R2, class C >
using CauchyFunction = Inv<Add<IntConstant<1>,Scal<C,R2>>>;

template < class R2, class C >
using LaplaceFunction = Exp<Minus< Scal<C,Sqrt<R2>>>>;

template < class R2, class C >
using InverseMultiquadricFunction = Inv<Sqrt<Add< Inv<C>,R2>>>;

template < class R2, class C, class W >
using SumGaussFunction = Scalprod<W,Exp<Scal<Minus<R2>,C>>>;

//////////////////////////////////////////////////////////////
////                 SCALAR RADIAL KERNELS                ////
//////////////////////////////////////////////////////////////

// Utility function

// for some reason the following variadic template version shoudl work but the nvcc compiler does not like it :
//template < class X, class Y, class B, template<class,class...> class F, class... PARAMS >
//using ScalarRadialKernel = Scal<F<SqDist<X,Y>,PARAMS...>,B>;

// so we use two distinct ScalarRadialKernel aliases, depending on the number of parameters :

template < class X, class Y, class B, template<class,class> class F, class PARAMS >
using ScalarRadialKernel_1 = Scal<F<SqDist<X,Y>,PARAMS>,B>;

template < class X, class Y, class B, template<class,class,class> class F, class PARAMS1, class PARAMS2 >
using ScalarRadialKernel_2 = Scal<F<SqDist<X,Y>,PARAMS1,PARAMS2>,B>;

// Utility aliases :
template < class C, class X, class Y, class B >
using GaussKernel = ScalarRadialKernel_1<X,Y,B,GaussFunction,C>;

template < class C, class X, class Y, class B >
using CauchyKernel = ScalarRadialKernel_1<X,Y,B,CauchyFunction,C>;

template < class C, class X, class Y, class B >
using LaplaceKernel = ScalarRadialKernel_1<X,Y,B,LaplaceFunction,C>;

template < class C, class X, class Y, class B >
using InverseMultiquadricKernel = ScalarRadialKernel_1<X,Y,B,InverseMultiquadricFunction,C>;

template < class C, class W, class X, class Y, class B >
using SumGaussKernel = ScalarRadialKernel_2<X,Y,B,SumGaussFunction,C,W>;

//////////////////////////////////////////////////////////////
////                 FACTORIZED GAUSS KERNEL              ////
//////////////////////////////////////////////////////////////
template < class C, class X, class Y, class B >
using GaussKernel_Factorized = Factorize< GaussKernel<C,X,Y,B> , Subtract<X,Y> >;


//////////////////////////////////////////////////////////////
////   DIRECT IMPLEMENTATIONS FOR SCALAR RADIAL KERNELS   ////
////	(FOR FASTER COMPUTATIONS)                         ////
//////////////////////////////////////////////////////////////

// specific implementation of the gaussian kernel and its gradient wrt to X


template < class C, class X, class Y, class B > struct GaussKernel_specific;
template < class C, class X, class Y, class B, class V, class GRADIN > struct GradGaussKernel_specific;

template < class C, class X, class Y, class B >
struct GaussKernel_specific {

    static_assert(C::DIM==1,"First template argument must be a of dimension 1 for GaussKernel_specific");
    static_assert(C::CAT==2,"First template argument must be a parameter variable (CAT=2) for GaussKernel_specific");
    static_assert(X::CAT!=Y::CAT,"Second and third template arguments must not be of the same category for GaussKernel_specific");
    static_assert(Y::CAT==B::CAT,"Third and fourth template arguments must be of the same category for GaussKernel_specific");

    using GenericVersion = GaussKernel<C,X,Y,B>;

    static const int DIM = GenericVersion::DIM;
    static const int DIMPOINT = X::DIM;
    static const int DIMVECT = DIM;

    static void PrintId(std::stringstream& str) {
        str << "GaussKernel_specific(";
        C::PrintId(str);
        str << ",";
        X::PrintId(str);
        str << ",";
        Y::PrintId(str);
        str << ",";
        B::PrintId(str);
        str << ")";
    }
    
    template < int CAT >
    using VARS = typename GenericVersion::template VARS<CAT>;

    using THIS = GaussKernel_specific<C,X,Y,B>;
		
    template<class U, class V>
    using Replace = CondType< V, THIS, IsSameType<U,THIS>::val >;
    
    using AllTypes = univpack<THIS>;

    template < class INDS, typename... ARGS >
    static HOST_DEVICE INLINE void Eval(__TYPE__* gammai, ARGS... args) {
        __TYPE__* params = Get<IndVal_Alias<INDS,C::N>::ind>(args...);
        __TYPE__* xi = Get<IndVal_Alias<INDS,X::N>::ind>(args...);
        __TYPE__* yj = Get<IndVal_Alias<INDS,Y::N>::ind>(args...);
        __TYPE__* betaj = Get<IndVal_Alias<INDS,B::N>::ind>(args...);
        __TYPE__ r2 = 0.0f;
        __TYPE__ temp;
        for(int k=0; k<DIMPOINT; k++) {
            temp =  yj[k]-xi[k];
            r2 += temp*temp;
        }
        __TYPE__ s = exp(-r2*params[0]);
        for(int k=0; k<DIMVECT; k++)
            gammai[k] = s * betaj[k];
    }

    template < class V, class GRADIN >
    using DiffT = GradGaussKernel_specific<C,X,Y,B,V,GRADIN>;

};

// by default we link to the standard autodiff versions of the gradients
template < class C, class X, class Y, class B, class V, class GRADIN >
struct GradGaussKernel_specific {
    using GenericVersion = Grad<GaussKernel<C,X,Y,B>,V,GRADIN>;

    static const int DIM = GenericVersion::DIM;
    static const int DIMPOINT = X::DIM;
    static const int DIMVECT = DIM;
	
	template < int CAT >
	using VARS = typename GenericVersion::template VARS<CAT>;
		
    static void PrintId(std::stringstream& str) {
        str << "GradGaussKernel_specific(";
        C::PrintId(str);
        str << ",";
        X::PrintId(str);
        str << ",";
        Y::PrintId(str);
        str << ",";
        B::PrintId(str);
        str << ",";
        V::PrintId(str);
        str << ",";
        GRADIN::PrintId(str);
        str << ")";
    }
    
    using THIS = GradGaussKernel_specific<C,X,Y,B,V,GRADIN>;
		
    template<class E, class F>
    using Replace = CondType< F, GradGaussKernel_specific<C,X,Y,B,V,typename GRADIN::template Replace<E,F>>, IsSameType<E,THIS>::val >;
    
    using AllTypes = MergePacks < univpack<THIS,V> , typename GRADIN::AllTypes >;

    template < class INDS, typename... ARGS >
    static HOST_DEVICE INLINE void Eval(__TYPE__* gammai, ARGS... args) {
        GenericVersion::template Eval<INDS>(gammai,args...);
    }

    template < class V2, class GRADIN2 >
    using DiffT = Grad<GenericVersion,V2,GRADIN2>;

};

// specific implementation of gradient wrt X
template < class C, class X, class Y, class B, class GRADIN >
struct GradGaussKernel_specific<C,X,Y,B,X,GRADIN> {
    using GenericVersion = Grad<GaussKernel<C,X,Y,B>,X,GRADIN>;

    static const int DIM = GenericVersion::DIM;
    static const int DIMPOINT = X::DIM;
    static const int DIMVECT = DIM;
	
	template < int CAT >
	using VARS = typename GenericVersion::template VARS<CAT>;
		
    using THIS = GradGaussKernel_specific<C,X,Y,B,X,GRADIN>;
		
    static void PrintId(std::stringstream& str) {
        str << "GradGaussKernel_specific(";
        C::PrintId(str);
        str << ",";
        X::PrintId(str);
        str << ",";
        Y::PrintId(str);
        str << ",";
        B::PrintId(str);
        str << ",";
        X::PrintId(str);
        str << ",";
        GRADIN::PrintId(str);
        str << ")";
    }
    
    template<class U, class V>
    using Replace = CondType< V, GradGaussKernel_specific<C,X,Y,B,X,typename GRADIN::template Replace<U,V>>, IsSameType<U,THIS>::val >;
    
    using AllTypes = MergePacks < univpack<THIS,X> , typename GRADIN::AllTypes >;

    template < class INDS, typename... ARGS >
    static HOST_DEVICE INLINE void Eval(__TYPE__* gammai, ARGS... args) {
        __TYPE__* params = Get<IndVal_Alias<INDS,C::N>::ind>(args...);
        __TYPE__* xi = Get<IndVal_Alias<INDS,X::N>::ind>(args...);
        __TYPE__* yj = Get<IndVal_Alias<INDS,Y::N>::ind>(args...);
        __TYPE__* betaj = Get<IndVal_Alias<INDS,B::N>::ind>(args...);
        __TYPE__* etai = Get<IndVal_Alias<INDS,GRADIN::N>::ind>(args...);

        __TYPE__ r2 = 0.0f, sga = 0.0f;          // Don't forget to initialize at 0.0
        __TYPE__ xmy[DIMPOINT];
        for(int k=0; k<DIMPOINT; k++) {      // Compute the L2 squared distance r2 = | x_i-y_j |_2^2
            xmy[k] =  xi[k]-yj[k];
            r2 += xmy[k]*xmy[k];
        }
        for(int k=0; k<DIMVECT; k++)         // Compute the L2 dot product <a_i, b_j>
            sga += betaj[k]*etai[k];
        __TYPE__ s = - 2.0 * sga * exp(-r2*params[0]);  // Don't forget the 2 !
        for(int k=0; k<DIMPOINT; k++)        // Increment the output vector gammai - which is a POINT
            gammai[k] = s * xmy[k];
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
template < template<class,class...> class FORTHO, template<class,class...> class FTILDE, class X, class Y, class B, class... PARAMS >
struct TRI_Kernel_helper
{
    static_assert(X::CAT!=Y::CAT,"Second and third template arguments must not be of the same category for TRI_Kernel");
    static_assert(Y::CAT==B::CAT,"Third and fourth template arguments must be of the same category for TRI_Kernel");
	using R2 = SqDist<X,Y>; 			// r2=|x-y|^2	
	using KORTHOR2 = FORTHO<R2,PARAMS...>;			// k_ortho(r2)
	using KTILDER2 = FTILDE<R2,PARAMS...>;			// k_tilde(r2)
	using XMY = Subtract<X,Y>;			// x-y
	using BDOTXMY = Scalprod<B,XMY>;			// <b,x-y>
	using D = Scalprod<BDOTXMY,XMY>;				// <b,x-y>(x-y)
	using type = Add<Scal<KORTHOR2,B>,Scal<KTILDER2,D>>;		// final formula 
	using factorized_type = Factorize<Factorize<type,R2>,XMY>;	// formula, factorized by r2 and x-y
};

// final definition is here
template < template<class,class...> class FORTHO, template<class,class...> class FTILDE, class X, class Y, class B, class... PARAMS >
using TRI_Kernel = typename TRI_Kernel_helper<FORTHO,FTILDE,X,Y,B,PARAMS...>::factorized_type;

// Div-free and curl-free kernel with gaussian functions.
// k_df(x,y)b = exp(-r^2/s2)*(((d-1)/(2c)-r^2)b + <b,x-y>(x-y))
// k_cf(x,y)b = exp(-r^2/s2)*(	   (1/(2c)) b   - <b,x-y>(x-y))
// The value of 1/s2 must be given as first parameter (_P<0,1>) when calling Eval()
// We do not use the previous template because exp(-r^2/s2) is factorized

template < class C, class X, class Y, class B >
struct DivFreeGaussKernel_helper
{
    static_assert(C::DIM==1,"First template argument must be a of dimension 1 for DivFreeGaussKernel");
    static_assert(C::CAT==2,"First template argument must be a parameter variable (CAT=2) for DivFreeGaussKernel");
    static_assert(X::CAT!=Y::CAT,"Second and third template arguments must not be of the same category for DivFreeGaussKernel");
    static_assert(X::DIM==Y::DIM,"Second and third template arguments must have the same dimensions for DivFreeGaussKernel");
    static_assert(Y::CAT==B::CAT,"Third and fourth template arguments must be of the same category for DivFreeGaussKernel");
    static const int DIM = X::DIM;
	using R2 = SqDist<X,Y>; 			// r2=|x-y|^2
	using XMY = Subtract<X,Y>;			// x-y
	using G = GaussFunction<R2,C>;				// exp(-r^2/s2)
	using TWOC = Scal<IntConstant<2>,C>;		 	// 2c
	using D1 = Divide<IntConstant<DIM-1>,TWOC>;			// (d-1)/(2c)
	using D2 = Scal<Subtract<D1,R2>,B>;				// ((d-1)/(2c)-r^2)b
	using BDOTXMY = Scalprod<B,XMY>;				// <b,x-y>
	using D = Scal<BDOTXMY,XMY>;					// <b,x-y>(x-y)
	using type = Scal<G,Add<D2,D>>;					// final formula
	using factorized_type = Factorize<Factorize<type,R2>,XMY>;	// formula, factorized by r2 and x-y
};

template < class C, class X, class Y, class B >
using DivFreeGaussKernel = typename DivFreeGaussKernel_helper<C,X,Y,B>::factorized_type;

template < class C, class X, class Y, class B >
struct CurlFreeGaussKernel_helper
{
    static_assert(C::DIM==1,"First template argument must be a of dimension 1 for CurlFreeGaussKernel");
    static_assert(C::CAT==2,"First template argument must be a parameter variable (CAT=2) for CurlFreeGaussKernel");
    static_assert(X::CAT!=Y::CAT,"Second and third template arguments must not be of the same category for CurlFreeGaussKernel");
    static_assert(X::DIM==Y::DIM,"Second and third template arguments must have the same dimensions for CurlFreeGaussKernel");
    static_assert(Y::CAT==B::CAT,"Third and fourth template arguments must be of the same category for CurlFreeGaussKernel");
    static const int DIM = X::DIM;
	using R2 = SqDist<X,Y>; 			// r2=|x-y|^2
	using XMY = Subtract<X,Y>;			// x-y
	using G = GaussFunction<R2,C>;				// exp(-r^2/s2)
	using TWOC = Scal<IntConstant<2>,C>;		 	// 2c
	using D1 = Divide<IntConstant<1>,TWOC>;			// 1/(2c)
	using D2 = Scal<D1,B>;						// (1/(2c))b
	using BDOTXMY = Scalprod<B,XMY>;				// <b,x-y>
	using D = Scal<BDOTXMY,XMY>;					// <b,x-y>(x-y)
	using type = Scal<G,Subtract<D2,D>>;				// final formula
	using factorized_type = Factorize<Factorize<type,R2>,XMY>;	// formula, factorized by r2 and x-y
};

template < class C, class X, class Y, class B >
using CurlFreeGaussKernel = typename CurlFreeGaussKernel_helper<C,X,Y,B>::factorized_type;

// Weighted combination of the two previous kernels, which gives a Translation and Rotation Invariant kernel with gaussian base function.
// k_tri(x,y)b = lambda * k_df(x,y)b + (1-lambda) * k_cf(x,y)b
// The weight lambda must be specified as the second parameter (_P<1>) when calling Eval()

template < class L, class C, class X, class Y, class B >
struct TRIGaussKernel_helper
{
    static_assert(L::DIM==1,"First template argument must be a of dimension 1 for TRIGaussKernel");
    static_assert(L::CAT==2,"First template argument must be a parameter variable (CAT=2) for TRIGaussKernel");
    static_assert(C::DIM==1,"Second template argument must be a of dimension 1 for TRIGaussKernel");
    static_assert(C::CAT==2,"Second template argument must be a parameter variable (CAT=2) for TRIGaussKernel");
    static_assert(X::CAT!=Y::CAT,"Third and fourth template arguments must not be of the same category for TRIGaussKernel");
    static_assert(X::DIM==Y::DIM,"Third and fourth template arguments must have the same dimensions for TRIGaussKernel");
    static_assert(Y::CAT==B::CAT,"Fourth and fifth template arguments must be of the same category for TRIGaussKernel");
    static const int DIM = X::DIM;
	using OML = Subtract<IntConstant<1>,L>;					// 1-lambda
	using DF = DivFreeGaussKernel_helper<C,X,Y,B>;				// k_df(x,y)b (the helper struct, because we need it below)
	using CF = CurlFreeGaussKernel_helper<C,X,Y,B>;				// k_cf(x,y)b (the helper struct, because we need it below)
	using type =  Add<Scal<L,typename DF::type>,Scal<OML,typename CF::type>>;	// final formula (not factorized)
	// here we can factorize a lot ; we look at common expressions in Div Free and Curl Free kernels:
	using G = typename DF::G;							// exp(-r^2/s2) can be factorized
	using D = typename DF::D;							// <b,x-y>(x-y) can be factorized
	using XMY = typename DF::XMY;						// x-y can be factorized
	using R2 = typename DF::R2;							// r^2 can be factorized
	using factorized_type = Factorize<Factorize<Factorize<Factorize<type,G>,D>,R2>,XMY>;
};
	
template < class L, class C, class X, class Y, class B >
using TRIGaussKernel = typename TRIGaussKernel_helper<L,C,X,Y,B>::factorized_type;
}
