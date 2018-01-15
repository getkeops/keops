/*
 *
 * The file where the elementary operators are defined.
 *
 * The core operators of our engine are :
 *      Var<N,DIM,CAT>				: the N-th variable, a vector of dimension DIM,
 *                                    with CAT = 0 (i-variable), 1 (j-variable) or 2 (parameter)
 *      Grad<F,V,GRADIN>			: gradient (in fact transpose of diff op) of F with respect to variable V, applied to GRADIN
 *      P<N>, or Param<N>			: the N-th parameter variable
 *      X<N,DIM>					: the N-th variable, vector of dimension DIM, CAT = 0
 *      Y<N,DIM>					: the N-th variable, vector of dimension DIM, CAT = 1
 *
 *
 * Available constants are :
 *
 *      Zero<DIM>					: zero-valued vector of dimension DIM
 *      IntConstant<N>				: constant integer function with value N
 *      Constant<PRM>				: constant function with value given by parameter PRM (ex : Constant<C> here)
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

#include "Pack.h"

#ifdef __CUDACC__
	#define INLINE __forceinline__
	#include <thrust/tuple.h>
	#define TUPLE_VERSION thrust
#else
	#define INLINE inline
	#define TUPLE_VERSION std
#endif

#include <tuple>
#include <cmath>

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
        using DIMSX = typename GetDims<VARSI>::template PUTLEFT<F::DIM>; // dimensions of "i" variables. We add the output's dimension.
        using DIMSY = GetDims<VARSJ>;                           // dimensions of "j" variables

        using FORM  = F;  // We need a way to access the actual function being used

        using INDSI = GetInds<VARSI>;
        using INDSJ = GetInds<VARSJ>;

        using INDS = ConcatPacks<INDSI,INDSJ>;  // indices of variables

        using tmp = typename F::template VARS<2>;
        static const int DIMPARAM = tmp::SIZE;

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
struct Var
{
    static const int N   = _N;   // The index and dimension of Var, formally specified using the
    static const int DIM = _DIM; // templating syntax, are accessible using Var::N, Var::DIM.
    static const int CAT = _CAT;

    static void PrintId() 
    {
    	cout << "Var<" << N << "," << DIM << "," << CAT << ">";
    }
    
    template<class A, class B>
    using Replace = Var<N,DIM,CAT>;
    
    using AllTypes = univpack<Var<N,DIM,CAT>>;

    template < int CAT_ >        // Var::VARS<1> = [Var(with CAT=0)] if Var::CAT=1, [] otherwise
    using VARS = CondType<univpack<Var<N,DIM,CAT>>,univpack<>,CAT==CAT_>;


    // Evaluate a variable given a list of arguments:
    //
    // Var( 5, DIM )::Eval< [ 2, 5, 0 ], type2, type5, type0 >( params, out, var2, var5, var0 )
    //
    // will see that the index 1 is targeted,
    // assume that "var5" is of size DIM, and copy its value in "out".
    template < class INDS, typename ...ARGS >
    static HOST_DEVICE INLINE void Eval(__TYPE__* params, __TYPE__* out, ARGS... args) {
        auto t = TUPLE_VERSION::make_tuple(args...); // let us access the args using indexing syntax
        // IndValAlias<INDS,N>::ind is the first index such that INDS[ind]==N. Let's call it "ind"
        __TYPE__* xi = TUPLE_VERSION::get<IndValAlias<INDS,N>::ind>(t); // xi = the "ind"-th argument.
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

// Unary and Binary structures for evaluation
// these will be used in the Eval member function of the math operations 
// (Add, Scal, Scalprod, Exp, etc., see files math.h and norms.h)
// we define them to be able to specialize evaluation when dealing with the Var class as template parameters
// in order to avoid the use of Eval member function of the Var class which does a useless vector copy.

// default unary operator class
template < class F, class FA >
struct UnaryOp {
	template < class INDS, typename... ARGS >
	static HOST_DEVICE INLINE void Eval(__TYPE__* params, __TYPE__* out, ARGS... args) {
		// we create a vector of size FA::DIM
        __TYPE__ outA[FA::DIM];
        // then we call the Eval function of FA
        FA::template Eval<INDS>(params,outA,args...);
        // then we call the Operation function
        F::Operation(out,outA);
    }
};

// specialization when template is of type Var
template < class F, int N, int DIM, int CAT >
struct UnaryOp<F,Var<N,DIM,CAT>> {
	template < class INDS, typename... ARGS >
	static HOST_DEVICE INLINE void Eval(__TYPE__* params, __TYPE__* out, ARGS... args) {
		// we do not need to create a vector ; just access the Nth argument of args
        auto t = TUPLE_VERSION::make_tuple(args...); 
        __TYPE__* outA = TUPLE_VERSION::get<IndValAlias<INDS,N>::ind>(t); // outA = the "ind"-th argument.
        // then we call the Operation function
        F::Operation(out,outA);
    }
};

// default binary operator class
template < class F, class FA, class FB >
struct BinaryOp {
	template < class INDS, typename... ARGS >
	static HOST_DEVICE INLINE void Eval(__TYPE__* params, __TYPE__* out, ARGS... args) {
		// we create vectors of sizes FA::DIM and FB::DIM
        __TYPE__ outA[FA::DIM], outB[FB::DIM];
        // then we call the Eval function of FA and FB
        FA::template Eval<INDS>(params,outA,args...);
        FB::template Eval<INDS>(params,outB,args...);
        // then we call the Operation function
        F::Operation(out,outA,outB);
    }
};

// specialization when left template is of type Var
template < class F, class FA, int N, int DIM, int CAT >
struct BinaryOp<F,FA,Var<N,DIM,CAT>> {
	template < class INDS, typename... ARGS >
	static HOST_DEVICE INLINE void Eval(__TYPE__* params, __TYPE__* out, ARGS... args) {
        // we create a vector and call Eval only for FA
        __TYPE__ outA[FA::DIM];
        FA::template Eval<INDS>(params,outA,args...);
        // access the Nth argument of args
        auto t = TUPLE_VERSION::make_tuple(args...); 
        __TYPE__* outB = TUPLE_VERSION::get<IndValAlias<INDS,N>::ind>(t); // outB = the "ind"-th argument.
        // then we call the Operation function
        F::Operation(out,outA,outB);
    }
};

// specialization when right template is of type Var
template < class F, class FB, int N, int DIM, int CAT >
struct BinaryOp<F,Var<N,DIM,CAT>,FB> {
	template < class INDS, typename... ARGS >
	static HOST_DEVICE INLINE void Eval(__TYPE__* params, __TYPE__* out, ARGS... args) {
		// we create a vector and call Eval only for FB
        __TYPE__ outB[FB::DIM];
        FB::template Eval<INDS>(params,outB,args...);
        // access the Nth argument of args
        auto t = TUPLE_VERSION::make_tuple(args...);
        __TYPE__* outA = TUPLE_VERSION::get<IndValAlias<INDS,N>::ind>(t); // outA = the "ind"-th argument.
        // then we call the Operation function
        F::Operation(out,outA,outB);
    }
};

// specialization when both templates are of type Var
template < class F, int NA, int DIMA, int CATA, int NB, int DIMB, int CATB >
struct BinaryOp<F,Var<NA,DIMA,CATA>,Var<NB,DIMB,CATB>> {
	template < class INDS, typename... ARGS >
	static HOST_DEVICE INLINE void Eval(__TYPE__* params, __TYPE__* out, ARGS... args) {
	 	// we access the NAth and NBth arguments of args
        auto t = TUPLE_VERSION::make_tuple(args...);
        __TYPE__* outA = TUPLE_VERSION::get<IndValAlias<INDS,NA>::ind>(t);
        __TYPE__* outB = TUPLE_VERSION::get<IndValAlias<INDS,NB>::ind>(t);
        // then we call the Operation function
        F::Operation(out,outA,outB);
    }
};


//////////////////////////////////////////////////////////////
////             N-th PARAMETER  : Param< N >             ////
//////////////////////////////////////////////////////////////

template < int N >
struct Param {
    static const int INDEX = N;
    static const int DIM = 1;
    
	template < int CAT >
    using VARS = CondType<univpack<Param<N>>,univpack<>,CAT==3>;
    
    static void PrintId() {
        cout << "Param<" << N << ">";
    }
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

template < int N >
using _P = Param<N>;


//////////////////////////////////////////////////////////////
////      FACTORIZE OPERATOR  : Factorize< F,G >          ////
//////////////////////////////////////////////////////////////

// Factorize< F,G > is the same as F, but when evaluating we factorize
// the computation of G, meaning that if G appears several times inside the
// formula F, we will compute it once only

template < class F, class G >
struct Factorize
{

    static const int DIM = F::DIM;
    
    static void PrintId() 
    {
    	cout << "Factorize<";
	F::PrintId();
	cout << ",";
	G::PrintId();
	cout << ">";
    }

    using THIS = Factorize<F,G>;    

    using Factor = G;

    // we define a new formula from F (called factorized formula), replacing G inside by a new variable ; this is used in function Eval()
    template < class INDS >
    using FactorizedFormula = typename F::template Replace<G,Var<INDS::SIZE,G::DIM,3>>;	// means replace G by Var<INDS::SIZE,G::DIM,3> in formula F

    template<class A, class B>
    using Replace = CondType< B , Factorize<typename F::template Replace<A,B>,typename G::template Replace<A,B>> , IsSameType<A,THIS>::val >;
        
    using AllTypes = MergePacks < MergePacks< univpack<THIS> , typename F::AllTypes > , typename G::AllTypes >;

    template < int CAT >       
    using VARS = typename F::template VARS<CAT>;

    template < class INDS, typename ...ARGS >
    static HOST_DEVICE INLINE void Eval(__TYPE__* params, __TYPE__* out, ARGS... args)
    {
	// First we compute G
	__TYPE__ outG[G::DIM];
	G::template Eval<INDS>(params,outG,args...);
	// Ffact is the factorized formula
	using Ffact = typename THIS::FactorizedFormula<INDS>;
	// new indices for the call to Eval : we add one more index to the list
	using NEWINDS = ConcatPacks<INDS,pack<INDS::SIZE>>;
	// call to Eval on the factorized formula, we pass outG as last parameter
	Ffact::template Eval<NEWINDS>(params,out,args...,outG);
    }
    
    template < class V, class GRADIN >
    using DiffT = Factorize<typename F::template DiffT<V,GRADIN>,G>;
    
};




//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
#include "formulas/constants.h"
#include "formulas/maths.h"
#include "formulas/norms.h"
#include "formulas/kernels.h"
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

