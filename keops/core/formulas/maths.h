#pragma once

#include <sstream>
#include <assert.h>

#include "core/Pack.h"
#include "core/autodiff.h"

#include "core/formulas/constants.h"

/*
 * The file where the elementary math operators are defined.
 * Available math operations are :
 *
 *   +, *, - :
 *      Add<FA,FB>					: adds FA and FB functions
 *      Subtract<FA,FB>					: subtractss FA and FB functions
 *      Scal<FA,FB>                 : product of FA (scalar valued) with FB
 *      Mult<FA,FB>                 : element-wise multiplication of FA and FB
 *      Minus<F>					: alias for Scal<IntConstant<-1>,F>
 *
 *   /, ^, ^2, ^-1, ^(1/2) :
 *      Divide<FA,FB>				: alias for Scal<FA,Inv<FB>>
 *      Pow<F,M>					: Mth power of F (vectorized) ; M is an integer
 *      Powf<A,B>					: alias for Exp<Scal<FB,Log<FA>>>
 *      Square<F>					: Pointwise square, more efficient than Pow<F,2>
 *      Inv<F>						: Pointwise inverse, more efficient than Pow<F,-1>
 *      IntInv<N>					: alias for Inv<IntConstant<N>>
 *      Sqrt<F>						: alias for Powf<F,IntInv<2>>
 *      Rsqrt<F>					: inverse square root
 *
 *   standard math functions :
 *      Exp<F>						: exponential of F (vectorized)
 *      Log<F>						: logarithm   of F (vectorized)
 *      Sin<F>						: sine        of F (vectorized)
 *      Cos<F>						: cosine      of F (vectorized)
 *      Sign<F>						: sign        of F (vectorized)
 *      Step<F>						: step        of F (vectorized)
 *      ReLU<F>						: ReLU        of F (vectorized)
 *      Sign<F>						: sign        of F (vectorized)
 *
 *   concatenation and matrix-vector products:
 *      Concat<FA,FB>				: concatenation of FB and FB
 *      MatVecMult<FA,FB>			: matrix-vector product (FA::DIM must be a muliple of FB::DIM)
 *      VecMatMult<FA,FB>			: vector-matrix product (FB::DIM must be a muliple of FA::DIM)
 *      TensorProd<FA,FB>			: tensor product (output is of dimension FA::DIM*FB::DIM)
 *
 */

namespace keops {

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

// Addition, Subtraction, Scalar product and "Scalar*Vector product" symbolic operators.
// The actual implementation can be found below.
// Since the gradients of these operations are "bootstrapped", we need to be a little bit
// careful with the declaration order, and therefore use three "typenames" per operation:
// Op_Alias, Op_Impl and Op (proper).
template < class FA, class FB > struct Add_Impl;
template < class FA, class FB > struct Subtract_Impl;
template < class FA, class FB > struct Scalprod_Impl;
template < class FA, class FB > struct Scal_Impl;
template < class FA, class FB > struct Mult_Impl;

template < class FA, class FB > struct Add_Alias;
template < class FA, class FB > struct Subtract_Alias;
template < class FA, class FB > struct Scalprod_Alias;
template < class FA, class FB > struct Scal_Alias;
template < class FA, class FB > struct Mult_Alias;
template < class F > struct Norm2_Alias;

template < class FA, class FB >
using Add = typename Add_Alias<FA,FB>::type;

template < class FA, class FB >
using Subtract = typename Subtract_Alias<FA,FB>::type;

template < class FA, class FB >
using Scalprod = typename Scalprod_Alias<FA,FB>::type;

template < class FA, class FB >
using Scal = typename Scal_Alias<FA,FB>::type;

template < class FA, class FB >
using Mult = typename Mult_Alias<FA,FB>::type;

//////////////////////////////////////////////////////////////
////               MINUS OPERATOR : Minus< F >            ////
//////////////////////////////////////////////////////////////

template < class F >
struct Minus : UnaryOp<Minus,F> {
    
    static const int DIM = F::DIM;

    static void PrintIdString(std::stringstream& str) { str << "Minus"; }
	
    static HOST_DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outF) {
         for(int k=0; k<DIM; k++)
             out[k] = -outF[k];
	}

    template < class V, class GRADIN >
    using DiffT = typename F::template DiffT<V,Minus<GRADIN>>;

};

//////////////////////////////////////////////////////////////
////               ADDITION : Add< FA,FB >                ////
//////////////////////////////////////////////////////////////


template < class FA, class FB >
struct Add_Impl : BinaryOp<Add_Impl,FA,FB> {
    // Output dim = FA::DIM = FB::DIM
    static const int DIM = FA::DIM;
    static_assert(DIM==FB::DIM,"Dimensions must be the same for Add");
    
    static void PrintIdString(std::stringstream& str) { str << "+"; }
    
    static HOST_DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outA, __TYPE__ *outB) {
            for(int k=0; k<DIM; k++)
            	out[k] = outA[k] + outB[k];
	}

    // [\partial_V (A + B) ] . gradin = [\partial_V A ] . gradin  + [\partial_V B ] . gradin
    template < class V, class GRADIN >
    using DiffT = Add < typename FA::template DiffT<V,GRADIN> , typename FB::template DiffT<V,GRADIN> >;

};

// Simplification rules
// We have to divide rules into several stages
// to avoid conflicts

// third stage

// base class : this redirects to the implementation
template < class FA, class FB >
struct Add_Alias0 {
    using type = Add_Impl<FA,FB>;
};

// A + A = 2A
template < class F >
struct Add_Alias0<F,F> {
    using type = Scal<IntConstant<2>,F>;
};

// A + B*A = (1+B)*A
template < class F, class G >
struct Add_Alias0<F,Scal_Impl<G,F>> {
    using type = Scal<Add<IntConstant<1>,G>,F>;
};

// B*A + A = (1+B)*A
template < class F, class G >
struct Add_Alias0<Scal_Impl<G,F>,F> {
    using type = Scal<Add<IntConstant<1>,G>,F>;
};

// second stage

// base class : this redirects to the third stage
template < class FA, class FB >
struct Add_Alias1 {
    using type = typename Add_Alias0<FA,FB>::type;
};

// B*A + C*A = (B+C)*A
template < class F, class G, class H >
struct Add_Alias1<Scal_Impl<G,F>,Scal_Impl<H,F>> {
    using type = Scal<Add<G,H>,F>;
};

// A+n = n+A (brings integers constants to the left)
template < int N, class F >
struct Add_Alias1<F,IntConstant_Impl<N>> {
    using type = Add<IntConstant<N>,F>;
};

// first stage

// base class : this redirects to the second stage
template < class FA, class FB >
struct Add_Alias {
    using type = typename Add_Alias1<FA,FB>::type;
};

// A + 0 = A
template < class FA, int DIM >
struct Add_Alias<FA,Zero<DIM>> {
    static_assert(DIM==FA::DIM,"Dimensions must be the same for Add");
    using type = FA;
};

// 0 + B = B
template < class FB, int DIM >
struct Add_Alias<Zero<DIM>,FB> {
    static_assert(DIM==FB::DIM,"Dimensions must be the same for Add");
    using type = FB;
};

// 0 + 0 = la tete a Toto
template < int DIM1, int DIM2 >
struct Add_Alias<Zero<DIM1>,Zero<DIM2>> {
    static_assert(DIM1==DIM2,"Dimensions must be the same for Add");
    using type = Zero<DIM1>;
};

// m+n = m+n
template < int M, int N >
struct Add_Alias<IntConstant_Impl<M>,IntConstant_Impl<N>> {
    using type = IntConstant<M+N>;
};


//////////////////////////////////////////////////////////////
////     VECTOR CONCATENATION : Concat<F,G>               ////
//////////////////////////////////////////////////////////////

template < class F, class G >
struct Concat_Impl : BinaryOp<Concat_Impl,F,G> {
    static const int DIM = F::DIM+G::DIM;
    
    static void PrintId(std::stringstream& str) { str << "Concat"; }

    static HOST_DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outF, __TYPE__ *outG) {
	    for(int k=0; k<F::DIM; k++)
            	out[k] = outF[k];
	    for(int k=0; k<G::DIM; k++)
            	out[k+F::DIM] = outG[k];
	}

    template < class V, class GRADIN >
    using DiffTF = typename F::template DiffT<V,GRADIN>;

    template < class V, class GRADIN >
    using DiffTG = typename G::template DiffT<V,GRADIN>;

    template < class V, class GRADIN >
    using DiffT = Add<DiffTF<V,Extract<GRADIN,0,F::DIM>>,DiffTG<V,Extract<GRADIN,F::DIM,DIM>>>;
};

template < class F, class G >
struct Concat_Alias {
	using type = Concat_Impl<F,G>;
};

// ugly stuff to make logsumexp reduction work
struct Dummy {
	static const int N = 0;
	static const int DIM = 0;
};

template < class F >
struct Concat_Alias<F,Dummy> {
	using type = F;
};

template < class F, class G >
using Concat = typename Concat_Alias<F,G>::type;

//////////////////////////////////////////////////////////////
////      Scal*Vector Multiplication : Scal< FA,FB>       ////
//////////////////////////////////////////////////////////////


template < class FA, class FB >
struct Scal_Impl : BinaryOp<Scal_Impl,FA,FB> {
    // FB is a vector, Output has the same size, and FA is a scalar
    static const int DIM = FB::DIM;
    static_assert(FA::DIM==1,"Dimension of FA must be 1 for Scal");

    static void PrintIdString(std::stringstream& str) { str << "*"; }

    static HOST_DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outA, __TYPE__ *outB) {
            for(int k=0; k<DIM; k++)
            	out[k] = *outA*outB[k];
	}

    //  \diff_V (A*B) = (\diff_V A) * B + A * (\diff_V B)
    // i.e.
    //  < \diff_V (A*B) . dV, gradin > = (\diff_V A).dV * <B,gradin> + A * < (\diff_V B).dV, gradin >
    //
    // so its L2 conjugate is given by :
    //
    // [\partial_V A*B] . gradin = [\partial_V A].(<gradin,B>) + A * [\partial_V B].gradin
    template < class V, class GRADIN >
    using DiffT = Add < typename FA::template DiffT<V,Scalprod<GRADIN,FB>> , Scal < FA, typename FB::template DiffT<V,GRADIN> > >;

};

// Simplification rules
// We have to divide rules into several stages
// to avoid conflicts

// second stage

// base class : this redirects to the implementation
template < class FA, class FB >
struct Scal_Alias0 {
    using type = Scal_Impl<FA,FB>;
};

// a*(b*c) = (a*b)*c
template < class FA, class F, class G >
struct Scal_Alias0<FA,Scal_Impl<F,G>> {
    using type = Scal<Scal<FA,F>,G>;
};

// m*n = m*n
template < int M, int N >
struct Scal_Alias0<IntConstant_Impl<M>,IntConstant_Impl<N>> {
    using type = IntConstant<M*N>;
};

// a*n = n*a
template < class FA, int N >
struct Scal_Alias0<FA,IntConstant_Impl<N>> {
    using type = Scal<IntConstant<N>,FA>;
};

// first stage

// base class : this redirects to the second stage
template < class FA, class FB >
struct Scal_Alias {
    using type = typename Scal_Alias0<FA,FB>::type;
};

// A * 0 = 0
template < class FA, int DIM >
struct Scal_Alias<FA,Zero<DIM>> {
    static_assert(1==FA::DIM,"Dimension of FA must be 1 for Scal");
    using type = Zero<DIM>;
};

// 0 * B = 0
template < class FB, int DIM >
struct Scal_Alias<Zero<DIM>,FB> {
    static_assert(DIM==1,"Dimension of FA must be 1 for Scal");
    using type = Zero<FB::DIM>;
};

// 0 * 0 = 0 (we have to specify it otherwise there is a conflict between A*0 and 0*B)
template < int DIM1, int DIM2 >
struct Scal_Alias<Zero<DIM1>,Zero<DIM2>> {
    static_assert(DIM1==1,"Dimension of FA must be 1 for Scal");
    using type = Zero<DIM2>;
};


//////////////////////////////////////////////////////////////
////      Element-wise Multiplication : Mult< FA,FB>      ////
//////////////////////////////////////////////////////////////


template < class FA, class FB >
struct Mult_Impl : BinaryOp<Mult_Impl,FA,FB> {
    // FA and FB are vectors with same size, Output has the same size
    static const int DIM = FA::DIM;
    static_assert(FA::DIM==DIM,"Dimensions of FA and FB must be the same for Mult");
    
    static void PrintIdString(std::stringstream& str) { str << "*"; }
    
    static HOST_DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outA, __TYPE__ *outB) {
        for(int k=0; k<DIM; k++)
            out[k] = outA[k]*outB[k];
    }
    
    //  \diff_V (A*B) = (\diff_V A) * B + A * (\diff_V B)
    template < class V, class GRADIN >
    using DiffTFA = typename FA::template DiffT<V,GRADIN>;
    
    template < class V, class GRADIN >
    using DiffTFB = typename FB::template DiffT<V,GRADIN>;

    template < class V, class GRADIN >
    using DiffT = Add<DiffTFA<V,Mult<FB,GRADIN>>,DiffTFB<V,Mult<FA,GRADIN>>>;
    
};

// Simplification rules

// base class : this redirects to the implementation
template < class FA, class FB >
struct Mult_Alias {
    using type = Mult_Impl<FA,FB>;
};

// A * 0 = 0
template < class FA, int DIM >
struct Mult_Alias<FA,Zero<DIM>> {
    static_assert(DIM==FA::DIM,"Dimensions of FA and FB must be the same for Mult");
    using type = Zero<DIM>;
};

// 0 * B = 0
template < class FB, int DIM >
struct Mult_Alias<Zero<DIM>,FB> {
    static_assert(DIM==FB::DIM,"Dimensions of FA and FB must be the same for Mult");
    using type = Zero<DIM>;
};

// 0 * 0 = 0 (we have to specify it otherwise there is a conflict between A*0 and 0*B)
template < int DIM1, int DIM2 >
struct Mult_Alias<Zero<DIM1>,Zero<DIM2>> {
    static_assert(DIM1==DIM2,"Dimensions of FA and FB must be the same for Mult");
    using type = Zero<DIM1>;
};


// small hack to be able to use the * operator for both 
// Scal and Mult depending on dimension in the new syntax

template < class FA, class FB >
using ScalOrMult = CondType<Scal<FA,FB>,Mult<FA,FB>,FA::DIM==1>;




//////////////////////////////////////////////////////////////
////             SUBTRACT : F-G		                      ////
//////////////////////////////////////////////////////////////


template < class FA, class FB >
struct Subtract_Impl : BinaryOp<Subtract_Impl,FA,FB> {
    // Output dim = FA::DIM = FB::DIM
    static const int DIM = FA::DIM;
    static_assert(DIM==FB::DIM,"Dimensions must be the same for Subtract");
    
    static void PrintIdString(std::stringstream& str) { str << "-"; }
    
    static HOST_DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outA, __TYPE__ *outB) {
            for(int k=0; k<DIM; k++)
            	out[k] = outA[k] - outB[k];
	}

    // [\partial_V (A - B) ] . gradin = [\partial_V A ] . gradin  - [\partial_V B ] . gradin
    template < class V, class GRADIN >
    using DiffT = Subtract < typename FA::template DiffT<V,GRADIN> , typename FB::template DiffT<V,GRADIN> >;

};

// Simplification rules

// third stage

// base class : this redirects to the implementation
template < class FA, class FB >
struct Subtract_Alias0 {
    using type = Subtract_Impl<FA,FB>;
};

// A - A = 0
template < class F >
struct Subtract_Alias0<F,F> {
    using type = Zero<F::DIM>;
};

// A - B*A = (1-B)*A
template < class F, class G >
struct Subtract_Alias0<F,Scal_Impl<G,F>> {
    using type = Scal<Subtract<IntConstant<1>,G>,F>;
};

// B*A - A = (-1+B)*A
template < class F, class G >
struct Subtract_Alias0<Scal_Impl<G,F>,F> {
    using type = Scal<Add<IntConstant<-1>,G>,F>;
};

// second stage

// base class : this redirects to third stage
template < class FA, class FB >
struct Subtract_Alias1 {
    using type = typename Subtract_Alias0<FA,FB>::type;
};

// B*A - C*A = (B-C)*A
template < class F, class G, class H >
struct Subtract_Alias1<Scal_Impl<G,F>,Scal_Impl<H,F>> {
    using type = Scal<Subtract<G,H>,F>;
};

// A-n = -n+A (brings integers constants to the left)
template < int N, class F >
struct Subtract_Alias1<F,IntConstant_Impl<N>> {
    using type = Add<IntConstant<-N>,F>;
};

// first stage

// base class, redirects to second stage
template < class FA, class FB >
struct Subtract_Alias {
    using type = typename Subtract_Alias1<FA,FB>::type;
};

// A - 0 = A
template < class FA, int DIM >
struct Subtract_Alias<FA,Zero<DIM>> {
    static_assert(DIM==FA::DIM,"Dimensions must be the same for Subtract");
    using type = FA;
};

// 0 - B = -B
template < class FB, int DIM >
struct Subtract_Alias<Zero<DIM>,FB> {
    static_assert(DIM==FB::DIM,"Dimensions must be the same for Subtract");
    using type = Minus<FB>;
};

// 0 - 0 = la tete a Toto
template < int DIM1, int DIM2 >
struct Subtract_Alias<Zero<DIM1>,Zero<DIM2>> {
    static_assert(DIM1==DIM2,"Dimensions must be the same for Subtract");
    using type = Zero<DIM1>;
};

// m-n = m-n
template < int M, int N >
struct Subtract_Alias<IntConstant_Impl<M>,IntConstant_Impl<N>> {
    using type = IntConstant<M-N>;
};



//////////////////////////////////////////////////////////////
////             EXPONENTIAL : Exp< F >                   ////
//////////////////////////////////////////////////////////////

template < class F >
struct Exp : UnaryOp<Exp,F> {
    
    static const int DIM = F::DIM;

    static void PrintIdString(std::stringstream& str) { str << "Exp"; }
	
    static HOST_DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outF) {
         for(int k=0; k<DIM; k++)
             out[k] = exp(outF[k]);
	}

    // [\partial_V exp(F)].gradin = exp(F) * [\partial_V F].gradin
    template < class V, class GRADIN >
    using DiffT = typename F::template DiffT<V,Mult<Exp<F>,GRADIN>>;

};

//////////////////////////////////////////////////////////////
////        SINE and COSINE : Sin< F >, Cos< F >          ////
//////////////////////////////////////////////////////////////

template < class F > struct Sin;
template < class F > struct Cos;

template < class F >
struct Sin : UnaryOp<Sin,F> {
    
    static const int DIM = F::DIM;

    static void PrintIdString(std::stringstream& str) { str << "Sin"; }
	
    static HOST_DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outF) {
         for(int k=0; k<DIM; k++)
             out[k] = sin(outF[k]);
	}

    template < class V, class GRADIN >
    using DiffT = typename F::template DiffT<V,Mult<Cos<F>,GRADIN>>;

};

template < class F >
struct Cos : UnaryOp<Cos,F> {
    
    static const int DIM = F::DIM;

    static void PrintIdString(std::stringstream& str) { str << "Cos"; }
	
    static HOST_DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outF) {
         for(int k=0; k<DIM; k++)
             out[k] = cos(outF[k]);
	}

    template < class V, class GRADIN >
    using DiffT = typename F::template DiffT<V,Minus<Mult<Sin<F>,GRADIN>>>;

};

//////////////////////////////////////////////////////////////
////             POWER OPERATOR : Pow< F, M >             ////
//////////////////////////////////////////////////////////////

template < class F, int M >
struct Pow : UnaryOp<Pow,F,M>  {
    
    static const int DIM = F::DIM;

    static void PrintIdString(std::stringstream& str) { str << "Pow"; }

    static HOST_DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outF) {
        for(int k=0; k<DIM; k++)
            out[k] = pow(outF[k],M);
	}

    // [\partial_V F^M].gradin  =  M * (F^(M-1)) * [\partial_V F].gradin
    template < class V, class GRADIN >
    using DiffTF = typename F::template DiffT<V,GRADIN>;

    template < class V, class GRADIN >
    using DiffT = Scal<IntConstant<M>,DiffTF<V,Mult<Pow<F,M-1>,GRADIN>>>;

};

//////////////////////////////////////////////////////////////
////             SQUARED OPERATOR : Square< F >           ////
//////////////////////////////////////////////////////////////

//template < class F >
//using Square = Pow<F,2>;

template < class F >
struct Square : UnaryOp<Square,F> {
    
    static const int DIM = F::DIM;

    static void PrintIdString(std::stringstream& str) { str << "Sq"; }
	
    static HOST_DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outF) {
         for(int k=0; k<DIM; k++) {
             __TYPE__ temp = outF[k];
             out[k] = temp*temp;
         }
	}

    template < class V, class GRADIN >
    using DiffTF = typename F::template DiffT<V,GRADIN>;

    // [\partial_V (F)**2].gradin = F * [\partial_V F].gradin
    template < class V, class GRADIN >
    using DiffT = Scal<IntConstant<2>,DiffTF<V,Mult<F,GRADIN>>> ;

};

//////////////////////////////////////////////////////////////
////      INVERSE : Inv<F>                                ////
//////////////////////////////////////////////////////////////

//template < class F >
//using Inv = Pow<F,-1>;

template < class F >
struct Inv : UnaryOp<Inv,F> {
    
    static const int DIM = F::DIM;

    static void PrintIdString(std::stringstream& str) { str << "Inv"; }
	
    static HOST_DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outF) {
         for(int k=0; k<DIM; k++) {
             out[k] = 1 / outF[k];
         }
}

    template < class V, class GRADIN >
    using DiffTF = typename F::template DiffT<V,GRADIN>;

    // [\partial_V (F)**2].gradin = F * [\partial_V F].gradin
    template < class V, class GRADIN >
    using DiffT = Scal<IntConstant<-1>,DiffTF<V,Mult<  Square<Inv<F>>  ,GRADIN>>> ;

};



//////////////////////////////////////////////////////////////
////      INVERSE OF INTEGER CONSTANT : Inv<N> is 1/N     ////
//////////////////////////////////////////////////////////////

// remark : there is currently no way to get a fixed real number directly...

template < int N >
using IntInv = Inv<IntConstant<N>>;

//////////////////////////////////////////////////////////////
////      DIVIDE : Divide<A,B> is A/B                     ////
//////////////////////////////////////////////////////////////

template < class FA, class FB >
using Divide = Scal<FA,Inv<FB>>;


//////////////////////////////////////////////////////////////
////             LOGARITHM : Log< F >                     ////
//////////////////////////////////////////////////////////////

template < class F >
struct Log : UnaryOp<Log,F> {
    static const int DIM = F::DIM;

    static void PrintIdString(std::stringstream& str) { str << "Log"; }

    static HOST_DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outF) {
        for(int k=0; k<DIM; k++)
            out[k] = log(outF[k]);
	}

    template < class V, class GRADIN >
    using DiffTF = typename F::template DiffT<V,GRADIN>;

    template < class V, class GRADIN >
    using DiffT = DiffTF<V,Mult<Inv<F>,GRADIN>>;
};

//////////////////////////////////////////////////////////////
////             SIGN : Sign< F >                         ////
//////////////////////////////////////////////////////////////

template < class F >
struct Sign : UnaryOp<Sign,F> {
    static const int DIM = F::DIM;

    static void PrintIdString(std::stringstream& str) { str << "Sign"; }

    static HOST_DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outF) {
        for(int k=0; k<DIM; k++)
			if(outF[k]<0)
            	out[k] = -1.0;
			else if(outF[k]==0)
				out[k] = 0.0;
			else
				out[k] = 1.0;
	}

    template < class V, class GRADIN >
    using DiffT = Zero<V::DIM>;
};

//////////////////////////////////////////////////////////////
////             STEP : Step< F >                         ////
//////////////////////////////////////////////////////////////

template < class F >
struct Step : UnaryOp<Step,F> {
    static const int DIM = F::DIM;

    static void PrintIdString(std::stringstream& str) { str << "Step"; }

    static HOST_DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outF) {
        for(int k=0; k<DIM; k++)
			if(outF[k]<0)
            	out[k] = 0.0;
			else
				out[k] = 1.0;
	}

    template < class V, class GRADIN >
    using DiffT = Zero<V::DIM>;
};

//////////////////////////////////////////////////////////////
////             RELU : ReLU< F >                         ////
//////////////////////////////////////////////////////////////

template < class F >
struct ReLU : UnaryOp<ReLU,F> {
    static const int DIM = F::DIM;

    static void PrintIdString(std::stringstream& str) { str << "ReLU"; }

    static HOST_DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outF) {
        for(int k=0; k<DIM; k++)
			if(outF[k]<0)
            	out[k] = 0.0;
			else
				out[k] = outF[k];
	}

    template < class V, class GRADIN >
    using DiffTF = typename F::template DiffT<V,GRADIN>;

    template < class V, class GRADIN >
    using DiffT = DiffTF<V,Mult<Step<F>,GRADIN>>;
};

//////////////////////////////////////////////////////////////
////             POWER OPERATOR : Powf< A, B >            ////
//////////////////////////////////////////////////////////////

template < class FA, class FB >
using Powf = Exp<Scal<FB,Log<FA>>>;


//////////////////////////////////////////////////////////////
////       SQUARE ROOT : Sqrt< F >                        ////
//////////////////////////////////////////////////////////////

template < class F > struct Sqrt_Impl; 
template < class F > struct Sqrt_Alias; 
template < class F > 
using Sqrt = typename Sqrt_Alias<F>::type; 

template < class F > struct Rsqrt_Impl; 
template < class F > struct Rsqrt_Alias; 
template < class F > 
using Rsqrt = typename Rsqrt_Alias<F>::type; 

template < class F >
struct Sqrt_Impl : UnaryOp<Sqrt_Impl,F> {
    static const int DIM = F::DIM;

    static void PrintIdString(std::stringstream& str) { str << "Sqrt"; }

    static HOST_DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outF) {
        for(int k=0; k<DIM; k++) 
            out[k] = sqrt(outF[k]);
	}

    template < class V, class GRADIN >
    using DiffTF = typename F::template DiffT<V,GRADIN>;

    template < class V, class GRADIN >
    using DiffT = DiffTF<V,Mult< Scal<IntInv<2>,Rsqrt<F>> ,GRADIN>>;
};

// Simplification rule

// base class, redirects to implementation
template < class F > 
struct Sqrt_Alias { 
    using type = Sqrt_Impl<F>; 
}; 
 
// Sqrt(0) = 0 
template < int DIM > 
struct Sqrt_Alias<Zero<DIM>> { 
    using type = Zero<DIM>; 
}; 




//////////////////////////////////////////////////////////////
////       INVERSE SQUARE ROOT : Rsqrt< F >               ////
//////////////////////////////////////////////////////////////


template < class F >
struct Rsqrt_Impl : UnaryOp<Rsqrt_Impl,F> {
    static const int DIM = F::DIM;

    static void PrintIdString(std::stringstream& str) { str << "Rsqrt"; }

    static HOST_DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outF) {
        for(int k=0; k<DIM; k++) 
            if(outF[k]==0)
                out[k] = 0;  // warning !! value should be Inf at 0 but we put 0 instead. This is intentional...
            else
#ifdef __NVCC__
                out[k] = rsqrt(outF[k]); 
#else
                out[k] = 1.0/sqrt(outF[k]); // should use specific rsqrt implementation for cpp ..
#endif
    }

    template < class V, class GRADIN >
    using DiffTF = typename F::template DiffT<V,GRADIN>;

    template < class V, class GRADIN >
    using DiffT = DiffTF<V,Mult< Scal<IntInv<-2>,Pow<Rsqrt<F>,3>> ,GRADIN>>;
};

// Simplification rule
 
// base class, redirects to implementation
template < class F > 
struct Rsqrt_Alias { 
    using type = Rsqrt_Impl<F>; 
}; 
 
// Rsqrt(0) = 0   // warning !! Rsqrt(0) should be Inf but we put 0 instead. This is intentional...
template < int DIM > 
struct Rsqrt_Alias<Zero<DIM>> { 
    using type = Zero<DIM>; 
}; 



/////////////////////////////////////////////////////////////////////////
////      Matrix-vector product      A x b                           ////
/////////////////////////////////////////////////////////////////////////

template < class A, class B > struct TensorProd;
template < class A, class B > struct VecMatMult;

template < class A, class B >
struct MatVecMult : BinaryOp<MatVecMult,A,B> {
    // A is vector of size n*p, interpreted as matrix (column major), B is vector of size p, interpreted as column vector
	// output is vector of size n
	
    static_assert(A::DIM % B::DIM == 0,"Dimensions of A and B are not compatible for matrix-vector product");

    static const int DIM = A::DIM / B::DIM;
    
    static void PrintIdString(std::stringstream& str) { str << "x"; }
    
    static HOST_DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outA, __TYPE__ *outB) {
        for(int k=0; k<DIM; k++) {
        	out[k] = 0;
        	for(int l=0; l<B::DIM; l++)
            	out[k] += outA[l*DIM+k] * outB[l];
        }
    }
    
    template < class V, class GRADIN >
    using DiffTA = typename A::template DiffT<V,GRADIN>;
    
    template < class V, class GRADIN >
    using DiffTB = typename B::template DiffT<V,GRADIN>;

    template < class V, class GRADIN >
    using DiffT = Add<DiffTA<V,TensorProd<GRADIN,B>>,DiffTB<V,VecMatMult<GRADIN,A>>>;
    
};


/////////////////////////////////////////////////////////////////////////
////     Vector-matrix product           b x A                       ////
/////////////////////////////////////////////////////////////////////////

template < class B, class A >
struct VecMatMult : BinaryOp<VecMatMult,B,A> {
    // A is vector of size n*p, interpreted as matrix (column major), B is vector of size n, interpreted as row vector
	// output is vector of size p
	
    static_assert(A::DIM % B::DIM == 0,"Dimensions of A and B are not compatible for matrix-vector product");

    static const int DIM = A::DIM / B::DIM;
    
    static void PrintIdString(std::stringstream& str) { str << "x"; }
    
    static HOST_DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outA, __TYPE__ *outB) {
        int q = 0;
        for(int k=0; k<DIM; k++) {
        	out[k] = 0;
        	for(int l=0; l<B::DIM; l++, q++)
            	out[k] += outA[q] * outB[l];
        }
    }
    
    template < class V, class GRADIN >
    using DiffTA = typename A::template DiffT<V,GRADIN>;
    
    template < class V, class GRADIN >
    using DiffTB = typename B::template DiffT<V,GRADIN>;

    template < class V, class GRADIN >
    using DiffT = Add<DiffTA<V,TensorProd<B,GRADIN>>,DiffTB<V,MatVecMult<A,GRADIN>>>;
    
};


/////////////////////////////////////////////////////////////////////////
////      Tensor product      a x b^T                                ////
/////////////////////////////////////////////////////////////////////////

template < class A, class B >
struct TensorProd : BinaryOp<TensorProd,A,B> {
    // A is vector of size n, B is vector of size p, 
	// output is vector of size n*p
	
    static const int DIM = A::DIM * B::DIM;
    
    static void PrintIdString(std::stringstream& str) { str << "(x)"; }
    
    static HOST_DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outA, __TYPE__ *outB) {
        int q = 0;
        for(int k=0; k<A::DIM; k++) {
        	for(int l=0; l<B::DIM; l++, q++)
            	out[q] = outA[k] * outB[l];
        }
    }
    
    template < class V, class GRADIN >
    using DiffTA = typename A::template DiffT<V,GRADIN>;
    
    template < class V, class GRADIN >
    using DiffTB = typename B::template DiffT<V,GRADIN>;

    template < class V, class GRADIN >
    using DiffT = Add<DiffTA<V,MatVecMult<GRADIN,B>>,DiffTB<V,VecMatMult<A,GRADIN>>>;
    
};





// iterate replace operator (should be put somewhere else)

template < class F, class G, class PACK >
struct IterReplace_Impl {
	using CURR = typename F::template Replace<G,typename PACK::FIRST>;
	using type = typename IterReplace_Impl<F,G,typename PACK::NEXT>::type::template PUTLEFT<CURR>;
};

template < class F, class G >
struct IterReplace_Impl<F,G,univpack<>> {
	using type = univpack<>;
};

template < class F, class G, class PACK >
using IterReplace = typename IterReplace_Impl<F,G,PACK>::type;


//////////////////////////////////////////////////////////////////////////////////////////////
////      Standard basis of R^DIM : < (1,0,0,...) , (0,1,0,...) , ... , (0,...,0,1) >     ////
//////////////////////////////////////////////////////////////////////////////////////////////

template < int DIM, int I=0 >
struct StandardBasis_Impl {
	using EI = ElemT<IntConstant<1>,DIM,I>;
	using type = typename StandardBasis_Impl<DIM,I+1>::type::template PUTLEFT<EI>;
};

template < int DIM >
struct StandardBasis_Impl<DIM,DIM> {
	using type = univpack<>;
};

template < int DIM >
using StandardBasis = typename StandardBasis_Impl<DIM>::type;

/////////////////////////////////////////////////////////////////////////
////      Matrix of gradient operator (=transpose of jacobian)       ////
/////////////////////////////////////////////////////////////////////////


template < class F, class V >
struct GradMatrix_Impl {
	using IndsTempVars = GetInds<typename F::template VARS<3>>;
	using GRADIN = Var<1+IndsTempVars::MAX,F::DIM,3>;
 	using packGrads = IterReplace< Grad<F,V,GRADIN> , GRADIN , StandardBasis<F::DIM> >;
 	using type = IterBinaryOp<Concat,packGrads>;
};

template < class F, class V >
using GradMatrix = typename GradMatrix_Impl<F,V>::type;




}
