/*
 *
 * The file where the elementary operators are defined.
 *
 * The core operators of our engine are :
 *      Var<N,DIM,CAT>				: the N-th variable, a vector of dimension DIM,
 *                                    with CAT = 0 (i-variable), 1 (j-variable) or 2 (parameter)
 *      Grad<F,V,GRADIN>			: gradient (in fact transpose of diff op) of F with respect to variable V, applied to GRADIN
 *      _P<N>, or Param<N>			: the N-th parameter variable
 *      _X<N,DIM>				: the N-th variable, vector of dimension DIM, CAT = 0
 *      _Y<N,DIM>				: the N-th variable, vector of dimension DIM, CAT = 1
 *      Elem<F,N>				: Extract Nth element of F
 *
 *
 * Available constants are :
 *
 *      Zero<DIM>					: zero-valued vector of dimension DIM
 *      IntConstant<N>				: constant integer function with value N
 *
 * Available math operations are :
 *
 *   +, *, - :
 *      Add<FA,FB>					: adds FA and FB functions
 *      Scal<FA,FB>					: product of FA (scalar valued) with FB
 *      Minus<F>					: alias for Scal<IntConstant<-1>,F>
 *      Subtract<FA,FB>				: alias for Add<FA,Minus<FB>>
 *
 *   /, ^, ^2, ^-1, ^(1/2) :
 *      Divide<FA,FB>				: alias for Scal<FA,Inv<FB>>
 *      Pow<F,M>					: Mth power of F (scalar valued) ; M is an integer
 *      Powf<A,B>					: alias for Exp<Scal<FB,Log<FA>>>
 *      Square<F>					: alias for Pow<F,2>
 *      Inv<F>						: alias for Pow<F,-1>
 *      IntInv<N>					: alias for Inv<IntConstant<N>>
 *      Sqrt<F>						: alias for Powf<F,IntInv<2>>
 *
 *   exp, log :
 *      Exp<F>						: exponential of F (scalar valued)
 *      Log<F>						: logarithm   of F (scalar valued)
 *
 * Available norms and scalar products are :
 *
 *   < .,. >, | . |^2, | .-. |^2 :
 *      Scalprod<FA,FB> 			: scalar product between FA and FB
 *      SqNorm2<F>					: alias for Scalprod<F,F>
 *      SqDist<A,B>					: alias for SqNorm2<Subtract<A,B>>
 *
 * Available kernel operations are :
 *
 *      GaussKernel<OOS2,X,Y,Beta>	: Gaussian kernel, OOS2 = 1/s^2
 *
 */

#pragma once

#include <iostream>
#include <cmath>

#include "core/Pack.h"

using namespace std;

// Generic function, created from a formula F, and a tag which is equal:
// - to 0 if you do the summation over j (with i the index of the output vector),
// - to 1 if you do the summation over i (with j the index of the output vector).
//
template < class F, int tagI=0 >
class Generic {

    static const int tagJ = 1-tagI;

  public :

    struct sEval { // static wrapper
        using VARSI = typename F::template VARS<tagI>; // Use the tag to select the "parallel"  variable
        using VARSJ = typename F::template VARS<tagJ>; // Use the tag to select the "summation" variable
        using VARSP = typename F::template VARS<2>;

        using DIMSX = typename GetDims<VARSI>::template PUTLEFT<F::DIM>; // dimensions of "i" variables. We add the output's dimension.
        using DIMSY = GetDims<VARSJ>;                           // dimensions of "j" variables
        using DIMSP = GetDims<VARSP>;                           // dimensions of parameters variables

        using FORM  = F;  // We need a way to access the actual function being used

        using INDSI = GetInds<VARSI>;
        using INDSJ = GetInds<VARSJ>;
        using INDSP = GetInds<VARSP>;

        using INDS = ConcatPacks<ConcatPacks<INDSI,INDSJ>,INDSP>;  // indices of variables
        static_assert(CheckAllDistinct<INDS>::val,"Incorrect formula : at least two distinct variables have the same position index.");

        template < typename... Args >
        HOST_DEVICE INLINE void operator()(Args... args) {
            F::template Eval<INDS>(args...);
        }
    };

};

template < int DIM > struct Zero; // Declare Zero in the header, for IdOrZeroAlias. Implementation below.

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

// IdOrZero( Vref, V, Fun ) = FUN                   if Vref == V
//                            Zero (of size V::DIM) if Vref != V
template < class Vref, class V, class FUN >
struct IdOrZeroAlias {
    using type = Zero<V::DIM>;
};

template < class V, class FUN >
struct IdOrZeroAlias<V,V,FUN> {
    using type = FUN;
};

template < class Vref, class V, class FUN >
using IdOrZero = typename IdOrZeroAlias<Vref,V,FUN>::type;

//////////////////////////////////////////////////////////////
////                      VARIABLE                        ////
//////////////////////////////////////////////////////////////

/*
 * Class for base variable
 * It is the atomic block of our autodiff engine.
 * A variable is given by :
 * - an index number _N (is it x1i, x2i, x3i or ... ?)
 * - a dimension _DIM of the vector
 * - a category CAT, equal to 0 if Var is "a  parallel variable" xi,
 *                   equal to 1 if Var is "a summation variable" yj.
 */
template < int _N, int _DIM, int _CAT=0 >
struct Var {
    static const int N   = _N;   // The index and dimension of Var, formally specified using the
    static const int DIM = _DIM; // templating syntax, are accessible using Var::N, Var::DIM.
    static const int CAT = _CAT;

    static void PrintId() {
        if(CAT==0)
            cout << "x";
        else if(CAT==1)
            cout << "y";
        else if(CAT==2)
	    cout << "p";
	else
            cout << "z";
        cout << N;
    }

    template<class A, class B>
    using Replace = Var<N,DIM,CAT>;

    using AllTypes = univpack<Var<N,DIM,CAT>>;

    template < int CAT_ >        // Var::VARS<1> = [Var(with CAT=0)] if Var::CAT=1, [] otherwise
    using VARS = CondType<univpack<Var<N,DIM,CAT>>,univpack<>,CAT==CAT_>;

    // Evaluate a variable given a list of arguments:
    //
    // Var( 5, DIM )::Eval< [ 2, 5, 0 ], type2, type5, type0 >( out, var2, var5, var0 )
    //
    // will see that the index 1 is targeted,
    // assume that "var5" is of size DIM, and copy its value in "out".
    template < class INDS, typename ...ARGS >
    static HOST_DEVICE INLINE void Eval(__TYPE__* out, ARGS... args) {
        // IndValAlias<INDS,N>::ind is the first index such that INDS[ind]==N. Let's call it "ind"
        __TYPE__* xi = Get<IndValAlias<INDS,N>::ind>(args...); // xi = the "ind"-th argument.
        for(int k=0; k<DIM; k++) // Assume that xi and out are of size DIM,
            out[k] = xi[k];      // and copy xi into out.
    }


    // Assuming that the gradient wrt. Var is GRADIN, how does it affect V ?
    // Var::DiffT<V, grad_input> = grad_input   if V == Var (in the sense that it represents the same symb. var.)
    //                             Zero(V::DIM) otherwise
    template < class V, class GRADIN >
    using DiffT = IdOrZero<Var<N,DIM,CAT>,V,GRADIN>;

};

//////////////////////////////////////////////////////////////////////////////////////////////////
////                      Unary and Binary operations wrappers                                ////
//////////////////////////////////////////////////////////////////////////////////////////////////

// Unary and Binary structures for defining common methods for the math operations
// (Add, Scal, Scalprod, Exp, etc., see files math.h and norms.h)
// we define common methods in a base class
// and then define a derived class to be able to specialize the evaluation method
// when dealing with the Var class as template parameters
// in order to avoid the use of Eval member function of the Var class which does a useless vector copy.

// unary operator base class : common methods
template < template<class,int...> class OP, class F, int... NS >
struct UnaryOp_base {

    using THIS = OP<F,NS...>;

    static void PrintId() {
        THIS::PrintIdString();
        cout << "(";
        F::PrintId();
        pack<NS...>::PrintComma();
        pack<NS...>::PrintAll();
        cout << ")";
    }

    using AllTypes = MergePacks<univpack<THIS>,typename F::AllTypes>;

    template<class A, class B>
    using Replace = CondType< B , OP<typename F::template Replace<A,B>,NS...> , IsSameType<A,THIS>::val >;

    // Vars(OP(F)) = Vars(F)
    template < int CAT >
    using VARS = typename F::template VARS<CAT>;

};

// unary operator class : default Eval method
template < template<class,int...> class OP, class F, int... NS >
struct UnaryOp : UnaryOp_base<OP,F,NS...> {

    using THIS = OP<F,NS...>;

    template < class INDS, typename... ARGS >
    static HOST_DEVICE INLINE void Eval(__TYPE__* out, ARGS... args) {
        // we create a vector of size F::DIM
        __TYPE__ outA[F::DIM];
        // then we call the Eval function of F
        F::template Eval<INDS>(outA,args...);
        // then we call the Operation function
        THIS::Operation(out,outA);
    }
};

// specialization when template F is of type Var
template < template<class,int...> class OP, int N, int DIM, int CAT, int... NS >
struct UnaryOp<OP,Var<N,DIM,CAT>,NS...>  : UnaryOp_base<OP,Var<N,DIM,CAT>,NS...> {

    using THIS = OP<Var<N,DIM,CAT>,NS...>;

    template < class INDS, typename... ARGS >
    static HOST_DEVICE INLINE void Eval(__TYPE__* out, ARGS... args) {
        // we do not need to create a vector ; just access the Nth argument of args
        __TYPE__* outA = Get<IndValAlias<INDS,N>::ind>(args...); // outA = the "ind"-th argument.
        // then we call the Operation function
        THIS::Operation(out,outA);
    }
};



// binary operator class : common methods
template < template<class,class> class OP, class FA, class FB >
struct BinaryOp_base {

    using THIS = OP<FA,FB>;

    static void PrintId() {
        cout << "(";
        FA::PrintId();
        THIS::PrintIdString();
        FB::PrintId();
        cout << ")";
    }

    static void PrintFactorized() {
        PrintId();
    }

    using AllTypes = MergePacks<univpack<OP<FA,FB>>,MergePacks<typename FA::AllTypes,typename FB::AllTypes>>;

    template<class A, class B>
    using Replace = CondType< B , OP<typename FA::template Replace<A,B>,typename FB::template Replace<A,B>> , IsSameType<A,THIS>::val >;

    // Vars( OP<FA,FB> ) = Vars(FA) U Vars(FB), whatever the category
    template < int CAT >
    using VARS = MergePacks<typename FA::template VARS<CAT>,typename FB::template VARS<CAT>>;

};


// binary operator class : default Eval method
template < template<class,class> class OP, class FA, class FB >
struct BinaryOp : BinaryOp_base<OP,FA,FB> {

    using THIS = OP<FA,FB>;

    template < class INDS, typename... ARGS >
    static HOST_DEVICE INLINE void Eval(__TYPE__* out, ARGS... args) {
        // we create vectors of sizes FA::DIM and FB::DIM
        __TYPE__ outA[FA::DIM], outB[FB::DIM];
        // then we call the Eval function of FA and FB
        FA::template Eval<INDS>(outA,args...);
        FB::template Eval<INDS>(outB,args...);
        // then we call the Operation function
        THIS::Operation(out,outA,outB);
    }
};

// specialization when left template is of type Var
template < template<class,class> class OP, int N, int DIM, int CAT, class FB >
struct BinaryOp<OP,Var<N,DIM,CAT>,FB>  : BinaryOp_base<OP,Var<N,DIM,CAT>,FB> {

    using THIS = OP<Var<N,DIM,CAT>,FB>;

    template < class INDS, typename... ARGS >
    static HOST_DEVICE INLINE void Eval(__TYPE__* out, ARGS... args) {
        // we create a vector and call Eval only for FB
        __TYPE__ outB[FB::DIM];
        FB::template Eval<INDS>(outB,args...);
        // access the Nth argument of args
        __TYPE__* outA = Get<IndValAlias<INDS,N>::ind>(args...); // outA = the "ind"-th argument.
        // then we call the Operation function
        THIS::Operation(out,outA,outB);
    }
};

// specialization when right template is of type Var
template < template<class,class> class OP, class FA, int N, int DIM, int CAT >
struct BinaryOp<OP,FA,Var<N,DIM,CAT>>  : BinaryOp_base<OP,FA,Var<N,DIM,CAT>> {

    using THIS = OP<FA,Var<N,DIM,CAT>>;

    template < class INDS, typename... ARGS >
    static HOST_DEVICE INLINE void Eval(__TYPE__* out, ARGS... args) {
        // we create a vector and call Eval only for FA
        __TYPE__ outA[FA::DIM];
        FA::template Eval<INDS>(outA,args...);
        // access the Nth argument of args
        __TYPE__* outB = Get<IndValAlias<INDS,N>::ind>(args...); // outB = the "ind"-th argument.
        // then we call the Operation function
        THIS::Operation(out,outA,outB);
    }
};

// specialization when both templates are of type Var
template < template<class,class> class OP, int NA, int DIMA, int CATA, int NB, int DIMB, int CATB >
struct BinaryOp<OP,Var<NA,DIMA,CATA>,Var<NB,DIMB,CATB>>  : BinaryOp_base<OP,Var<NA,DIMA,CATA>,Var<NB,DIMB,CATB>> {

    using THIS = OP<Var<NA,DIMA,CATA>,Var<NB,DIMB,CATB>>;

    template < class INDS, typename... ARGS >
    static HOST_DEVICE INLINE void Eval(__TYPE__* out, ARGS... args) {
        // we access the NAth and NBth arguments of args
        __TYPE__* outA = Get<IndValAlias<INDS,NA>::ind>(args...);
        __TYPE__* outB = Get<IndValAlias<INDS,NB>::ind>(args...);
        // then we call the Operation function
        THIS::Operation(out,outA,outB);
    }
};




//////////////////////////////////////////////////////////////
////     ELEMENT EXTRACTION : Elem<F,M>                   ////
//////////////////////////////////////////////////////////////

template < class F, int N, int M > struct ElemT;

template < class F, int M >
struct Elem : UnaryOp<Elem,F,M> {
    static const int DIM = 1;
    static_assert(F::DIM>M,"Index out of bound in Elem");

    static void PrintId() { cout << "Elem"; }

    static HOST_DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outF) {
            *out = outF[M];
	}

    template < class V, class GRADIN >
    using DiffTF = typename F::template DiffT<V,GRADIN>;

    template < class V, class GRADIN >
    using DiffT = DiffTF<V,ElemT<GRADIN,F::DIM,M>>;
};



//////////////////////////////////////////////////////////////
////     ELEMENT "INJECTION" : ElemT<F,N,M>               ////
//////////////////////////////////////////////////////////////

template < class F, int N, int M >
struct ElemT : UnaryOp<ElemT,F,N,M> {
    static const int DIM = N;
    static_assert(F::DIM==1,"Input of ElemT should be a scalar");

    static void PrintId() { cout << "ElemT"; }

    static HOST_DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outF) {
	    for(int k=0; k<DIM; k++)
            	out[k] = 0.0;
	    out[M] = *outF;
	}

    template < class V, class GRADIN >
    using DiffTF = typename F::template DiffT<V,GRADIN>;

    template < class V, class GRADIN >
    using DiffT = DiffTF<V,Elem<GRADIN,M>>;
};





// helper for counting the number of occurrences of a subformula in a formula

template<class F, class G>
struct CountIn_ {
    static const int val = 0;
};

template<class F>
struct CountIn_<F,F> {
    static const int val = 1;
};

template<class F, class G>
struct CountIn {
    static const int val = CountIn_<F,G>::val;
};

template<template<class,int...> class OP, class F, class G, int... NS>
struct CountIn<OP<F,NS...>,G> {
    static const int val = CountIn_<OP<F,NS...>,G>::val + CountIn<F,G>::val;
};

template<template<class,class> class OP, class FA, class FB, class G>
struct CountIn<OP<FA,FB>,G> {
    static const int val = CountIn_<OP<FA,FB>,G>::val + CountIn<FA,G>::val + CountIn<FB,G>::val;
};



//////////////////////////////////////////////////////////////
////      GRADIENT OPERATOR  : Grad< F, V, Gradin >       ////
//////////////////////////////////////////////////////////////

// Computes [\partial_V F].gradin
// Symbolic differentiation is a straightforward recursive operation,
// provided that the operators have implemented their DiffT "compiler methods":
template < class F, class V, class GRADIN >
using Grad = typename F::template DiffT<V,GRADIN>;


//////////////////////////////////////////////////////////////
////    STANDARD VARIABLES :_X<N,DIM>,_Y<N,DIM>,_P<N>     ////
//////////////////////////////////////////////////////////////

// N.B. : We leave "X", "Y" and "P" to the user
//        and restrict ourselves to "_X", "_Y", "_P".

template < int N, int DIM >
using _X = Var<N,DIM,0>;

template < int N, int DIM >
using _Y = Var<N,DIM,1>;

template < int N, int DIM >
using Param = Var<N,DIM,2>;

template < int N, int DIM >
using _P = Param<N,DIM>;


// Print formula to standard output

template < class F >
void PrintFormula() {
    cout << "Variables : ";
    using Vars0 = typename F::template VARS<0>;
    using Dims0 = GetDims<Vars0>;
    using Inds0 = GetInds<Vars0>;
    for(int k=0; k<Vars0::SIZE; k++)
        cout << "x" << Inds0::VAL(k) << " (dim=" << Dims0::VAL(k) << "), ";
    using Vars1 = typename F::template VARS<1>;
    using Dims1 = GetDims<Vars1>;
    using Inds1 = GetInds<Vars1>;
    for(int k=0; k<Vars1::SIZE; k++)
        cout << "y" << Inds1::VAL(k) << " (dim=" << Dims1::VAL(k) << "), ";
    using Vars2 = typename F::template VARS<2>;
    using Dims2 = GetDims<Vars2>;
    using Inds2 = GetInds<Vars2>;
    for(int k=0; k<Vars2::SIZE; k++)
        cout << "p" << Inds2::VAL(k) << " (dim=" << Dims2::VAL(k) << "), ";
    cout << endl;
    cout << "Formula = ";
    F::PrintId();
}


