#pragma once

#include <iostream>
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
 *      Scal<FA,FB>                 : product of FA (scalar valued) with FB
 *      Mult<FA,FB>                 : element-wise multiplication of FA and FB
 *      Minus<F>					: alias for Scal<IntConstant<-1>,F>
 *      Subtract<FA,FB>				: alias for Add<FA,Minus<FB>>
 *
 *   /, ^, ^2, ^-1, ^(1/2) :
 *      Divide<FA,FB>				: alias for Scal<FA,Inv<FB>>
 *      Pow<F,M>					: Mth power of F (vectorized) ; M is an integer
 *      Powf<A,B>					: alias for Exp<Scal<FB,Log<FA>>>
 *      Square<F>					: alias for Pow<F,2>
 *      Inv<F>						: alias for Pow<F,-1>
 *      IntInv<N>					: alias for Inv<IntConstant<N>>
 *      Sqrt<F>						: alias for Powf<F,IntInv<2>>
 *
 *   exp, log :
 *      Exp<F>						: exponential of F (vectorized)
 *      Log<F>						: logarithm   of F (vectorized)
 *
 */



//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

// Addition, Scalar product and "Scalar*Vector product" symbolic operators.
// The actual implementation can be found below.
// Since the gradients of these operations are "bootstrapped", we need to be a little bit
// careful with the declaration order, and therefore use three "typenames" per operation:
// OpAlias, OpImpl and Op (proper).
template < class FA, class FB > struct AddImpl;
template < class FA, class FB > struct ScalprodImpl;
template < class FA, class FB > struct ScalImpl;
template < class FA, class FB > struct MultImpl;

template < class FA, class FB > struct AddAlias;
template < class FA, class FB > struct ScalprodAlias;
template < class FA, class FB > struct ScalAlias;
template < class FA, class FB > struct MultAlias;
template < class F > struct Norm2Alias;

template < class FA, class FB >
using Add = typename AddAlias<FA,FB>::type;

template < class FA, class FB >
using Scalprod = typename ScalprodAlias<FA,FB>::type;

template < class FA, class FB >
using Scal = typename ScalAlias<FA,FB>::type;

template < class FA, class FB >
using Mult = typename MultAlias<FA,FB>::type;

//////////////////////////////////////////////////////////////
////               MINUS OPERATOR : Minus< F >            ////
//////////////////////////////////////////////////////////////

template < class F >
using Minus = Scal<IntConstant<-1>,F>;


//////////////////////////////////////////////////////////////
////               ADDITION : Add< FA,FB >                ////
//////////////////////////////////////////////////////////////


template < class FA, class FB >
struct AddImpl : BinaryOp<AddImpl,FA,FB> {
    // Output dim = FA::DIM = FB::DIM
    static const int DIM = FA::DIM;
    static_assert(DIM==FB::DIM,"Dimensions must be the same for Add");
    
    static void PrintIdString() { cout << "+"; }
    
    static HOST_DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outA, __TYPE__ *outB) {
            for(int k=0; k<DIM; k++)
            	out[k] = outA[k] + outB[k];
	}

    // [\partial_V (A + B) ] . gradin = [\partial_V A ] . gradin  + [\partial_V B ] . gradin
    template < class V, class GRADIN >
    using DiffT = Add < typename FA::template DiffT<V,GRADIN> , typename FB::template DiffT<V,GRADIN> >;

};

template < class FA, class FB >
struct AddAlias0 {
    using type = AddImpl<FA,FB>;
};

// A + A = 2A
template < class F >
struct AddAlias0<F,F> {
    using type = Scal<IntConstant<2>,F>;
};

// A + B*A = (1+B)*A
template < class F, class G >
struct AddAlias0<F,ScalImpl<G,F>> {
    using type = Scal<Add<IntConstant<1>,G>,F>;
};

// B*A + A = (1+B)*A
template < class F, class G >
struct AddAlias0<ScalImpl<G,F>,F> {
    using type = Scal<Add<IntConstant<1>,G>,F>;
};

template < class FA, class FB >
struct AddAlias1 {
    using type = typename AddAlias0<FA,FB>::type;
};

// B*A + C*A = (B+C)*A
template < class F, class G, class H >
struct AddAlias1<ScalImpl<G,F>,ScalImpl<H,F>> {
    using type = Scal<Add<G,H>,F>;
};

// A+n = n+A (brings integers constants to the left)
template < int N, class F >
struct AddAlias1<F,IntConstant<N>> {
    using type = Add<IntConstant<N>,F>;
};

template < class FA, class FB >
struct AddAlias {
    using type = typename AddAlias1<FA,FB>::type;
};

// Constants, etc. will lead to the creation of *many* zero vectors when computing the gradient.
// Even though this backpropagation engine makes few optimizations,
// this is definitely the one that should not be forgotten.

// A + 0 = A
template < class FA, int DIM >
struct AddAlias<FA,Zero<DIM>> {
    static_assert(DIM==FA::DIM,"Dimensions must be the same for Add");
    using type = FA;
};

// 0 + B = B
template < class FB, int DIM >
struct AddAlias<Zero<DIM>,FB> {
    static_assert(DIM==FB::DIM,"Dimensions must be the same for Add");
    using type = FB;
};

// 0 + 0 = la tete a Toto
template < int DIM1, int DIM2 >
struct AddAlias<Zero<DIM1>,Zero<DIM2>> {
    static_assert(DIM1==DIM2,"Dimensions must be the same for Add");
    using type = Zero<DIM1>;
};

// m+n = m+n
template < int M, int N >
struct AddAlias<IntConstant<M>,IntConstant<N>> {
    using type = IntConstant<M+N>;
};

//////////////////////////////////////////////////////////////
////      Scal*Vector Multiplication : Scal< FA,FB>       ////
//////////////////////////////////////////////////////////////


template < class FA, class FB >
struct ScalImpl : BinaryOp<ScalImpl,FA,FB> {
    // FB is a vector, Output has the same size, and FA is a scalar
    static const int DIM = FB::DIM;
    static_assert(FA::DIM==1,"Dimension of FA must be 1 for Scal");

    static void PrintIdString() { cout << "*"; }

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


template < class FA, class FB >
struct ScalAlias0 {
    using type = ScalImpl<FA,FB>;
};

template < class FA, class F, class G >
struct ScalAlias0<FA,ScalImpl<F,G>> {
    using type = Scal<Scal<FA,F>,G>;
};

template < int M, int N >
struct ScalAlias0<IntConstant<M>,IntConstant<N>> {
    using type = IntConstant<M*N>;
};

template < class FA, int N >
struct ScalAlias0<FA,IntConstant<N>> {
    using type = Scal<IntConstant<N>,FA>;
};

template < class FA, class FB >
struct ScalAlias {
    using type = typename ScalAlias0<FA,FB>::type;
};


// Constants, etc. will lead to the creation of *many* zero vectors when computing the gradient.
// Even though this backpropagation engine makes few optimizations,
// this is definitely the one that should not be forgotten.

// A * 0 = 0
template < class FA, int DIM >
struct ScalAlias<FA,Zero<DIM>> {
    static_assert(1==FA::DIM,"Dimension of FA must be 1 for Scal");
    using type = Zero<DIM>;
};

// 0 * B = 0
template < class FB, int DIM >
struct ScalAlias<Zero<DIM>,FB> {
    static_assert(DIM==1,"Dimension of FA must be 1 for Scal");
    using type = Zero<FB::DIM>;
};

// 0 * 0 = 0
template < int DIM1, int DIM2 >
struct ScalAlias<Zero<DIM1>,Zero<DIM2>> {
    static_assert(DIM1==1,"Dimension of FA must be 1 for Scal");
    using type = Zero<DIM2>;
};


//////////////////////////////////////////////////////////////
////      Element-wise Multiplication : Mult< FA,FB>      ////
//////////////////////////////////////////////////////////////


template < class FA, class FB >
struct MultImpl : BinaryOp<MultImpl,FA,FB> {
    // FA and FB are vectors with same size, Output has the same size
    static const int DIM = FA::DIM;
    static_assert(FA::DIM==DIM,"Dimensions of FA and FB must be the same for Mult");
    
    static void PrintIdString() { cout << "*"; }
    
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


template < class FA, class FB >
struct MultAlias {
    using type = MultImpl<FA,FB>;
};

// Constants, etc. will lead to the creation of *many* zero vectors when computing the gradient.
// Even though this backpropagation engine makes few optimizations,
// this is definitely the one that should not be forgotten.

// A * 0 = 0
template < class FA, int DIM >
struct MultAlias<FA,Zero<DIM>> {
    static_assert(DIM==FA::DIM,"Dimensions of FA and FB must be the same for Mult");
    using type = Zero<DIM>;
};

// 0 * B = 0
template < class FB, int DIM >
struct MultAlias<Zero<DIM>,FB> {
    static_assert(DIM==FB::DIM,"Dimensions of FA and FB must be the same for Mult");
    using type = Zero<DIM>;
};

// 0 * 0 = 0
template < int DIM1, int DIM2 >
struct MultAlias<Zero<DIM1>,Zero<DIM2>> {
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
using Subtract = Add<FA,Minus<FB>>;

//////////////////////////////////////////////////////////////
////             EXPONENTIAL : Exp< F >                   ////
//////////////////////////////////////////////////////////////

template < class F >
struct Exp : UnaryOp<Exp,F> {
    
    static const int DIM = F::DIM;

    static void PrintIdString() { cout << "Exp"; }
	
    static HOST_DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outF) {
         for(int k=0; k<DIM; k++)
             out[k] = exp(outF[k]);
	}

    // [\partial_V exp(F)].gradin = exp(F) * [\partial_V F].gradin
    template < class V, class GRADIN >
    using DiffT = typename F::template DiffT<V,Mult<Exp<F>,GRADIN>>;

};

//////////////////////////////////////////////////////////////
////             POWER OPERATOR : Pow< F, M >             ////
//////////////////////////////////////////////////////////////

template < class F, int M >
struct Pow : UnaryOp<Pow,F,M>  {
    
    static const int DIM = F::DIM;

    static void PrintIdString() { cout << "Pow"; }

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

template < class F >
using Square = Pow<F,2>;

//////////////////////////////////////////////////////////////
////      INVERSE : Inv<F>                                ////
//////////////////////////////////////////////////////////////

template < class F >
using Inv = Pow<F,-1>;

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

    static void PrintId() { cout << "Log"; }

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
////             POWER OPERATOR : Powf< A, B >            ////
//////////////////////////////////////////////////////////////

template < class FA, class FB >
using Powf = Exp<Scal<FB,Log<FA>>>;


//////////////////////////////////////////////////////////////
////       SQUARE ROOT : Sqrt< F >                        ////
//////////////////////////////////////////////////////////////

template < class F > struct SqrtImpl; 
template < class F > struct SqrtAlias; 
template < class F > 
using Sqrt = typename SqrtAlias<F>::type; 

template < class F > struct RsqrtImpl; 
template < class F > struct RsqrtAlias; 
template < class F > 
using Rsqrt = typename RsqrtAlias<F>::type; 

template < class F >
struct SqrtImpl : UnaryOp<SqrtImpl,F> {
    static const int DIM = F::DIM;

    static void PrintId() { cout << "Sqrt"; }

    static HOST_DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outF) {
        for(int k=0; k<DIM; k++) 
            out[k] = sqrt(outF[k]);
	}

    template < class V, class GRADIN >
    using DiffTF = typename F::template DiffT<V,GRADIN>;

    template < class V, class GRADIN >
    using DiffT = DiffTF<V,Mult< Scal<IntInv<2>,Rsqrt<F>> ,GRADIN>>;
};


template < class F > 
struct SqrtAlias { 
    using type = SqrtImpl<F>; 
}; 
 
// One simple optimization : 
 
// Sqrt(0) = 0 
template < int DIM > 
struct SqrtAlias<Zero<DIM>> { 
    using type = Zero<DIM>; 
}; 




//////////////////////////////////////////////////////////////
////       INVERSE SQUARE ROOT : Rsqrt< F >               ////
//////////////////////////////////////////////////////////////


template < class F >
struct RsqrtImpl : UnaryOp<RsqrtImpl,F> {
    static const int DIM = F::DIM;

    static void PrintId() { cout << "Rsqrt"; }

    static HOST_DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outF) {
        for(int k=0; k<DIM; k++) 
            if(outF[k]==0)
		out[k] = 0;  // warning !! value should be Inf at 0 but we put 0 instead. This is intentional...
	    else
                out[k] = 1.0/sqrt(outF[k]); // should use specific rsqrt implementation
	}

    template < class V, class GRADIN >
    using DiffTF = typename F::template DiffT<V,GRADIN>;

    template < class V, class GRADIN >
    using DiffT = DiffTF<V,Mult< Scal<IntInv<-2>,Pow<Rsqrt<F>,3>> ,GRADIN>>;
};


template < class F > 
struct RsqrtAlias { 
    using type = RsqrtImpl<F>; 
}; 
 
// One simple optimization : 
 
// Rsqrt(0) = 0   // warning !! Rsqrt(0) should be Inf but we put 0 instead. This is intentional...
template < int DIM > 
struct RsqrtAlias<Zero<DIM>> { 
    using type = Zero<DIM>; 
}; 



