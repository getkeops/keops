/*
 * 
 * The file where the elementary operators are defined.
 * Available operations are :
 *      IntConstant<N>				: constant integer function with value N
 *      Constant<PRM>				: constant function with value given by parameter PRM (ex : Constant<C> here)
 *      Add<FA,FB>					: adds FA and FB functions
 *      Scalprod<FA,FB> 			: scalar product between FA and FB
 *      Scal<FA,FB>					: product of FA (scalar valued) with FB
 *      SqNorm2<F>					: alias for Scalprod<F,F>
 *      Exp<F>						: exponential of F (scalar valued)
 *      Pow<F,M>					: Mth power of F (scalar valued) ; M is an integer
 *      Square<F>					: alias for Pow<F,2>
 *      Minus<F>					: alias for Scal<IntConstant<-1>,F>
 *      Subtract<FA,FB>				: alias for Add<FA,Minus<FB>>
 *      GaussKernel<PRM,FA,FB,FC> 	: alias for Scal<Exp<Scal<Constant<PRM>,Minus<SqNorm2<Subtract<FA,FB>>>>>,FC>
 *      Grad<F,V,GRADIN>			: gradient (in fact transpose of diff op) of F with respect to variable V, applied to GRADIN
 * 
 */

// N.B.: this file assumes that Pack.h has already been loaded, defining univpack and other collection class.

#define INLINE static __host__ __device__ __forceinline__ 
//#define INLINE static inline

#include <tuple>
#include <cmath>
#include <thrust/tuple.h>

using namespace std;


// At compilation time, detect the maximum between two values (typically, dimensions)
template <typename T>
static constexpr T static_max(T a, T b) 
{
    return a < b ? b : a;
}

template < int DIM > struct Zero; // Declare Zero in the header, for IdOrZeroAlias. Implementation below.

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

// IdOrZero( Vref, V, Fun ) = FUN                   if Vref == V
//                            Zero (of size V::DIM) if Vref != V
template < class Vref, class V, class FUN > 
struct IdOrZeroAlias
{   using type = Zero<V::DIM>; };

template < class V, class FUN >
struct IdOrZeroAlias<V,V,FUN>
{   using type = FUN; };

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
template < int _N, int _DIM, int CAT=0 >
struct Var
{
    static const int N   = _N;   // The index and dimension of Var, formally specified using the
    static const int DIM = _DIM; // templating syntax, are accessible using Var::N, Var::DIM.

    template < int CAT_ >        // Var::VARS<1> = [Var(with CAT=0)] if Var::CAT=1, [] otherwise
    using VARS = CondType<univpack<Var<N,DIM>>,univpack<>,CAT==CAT_>;

    // Evaluate a variable given a list of arguments:
    //
    // Var( 5, DIM )::Eval< [ 2, 5, 0 ], type2, type5, type0 >( params, out, var2, var5, var0 )
    // 
    // will see that the index 1 is targeted,
    // assume that "var5" is of size DIM, and copy its value in "out".
    template < class INDS, typename ...ARGS >
    INLINE void Eval(__TYPE__* params, __TYPE__* out, ARGS... args)
    {
        auto t = thrust::make_tuple(args...); // let us access the args using indexing syntax
        // IndValAlias<INDS,N>::ind is the first index such that INDS[ind]==N. Let's call it "ind"
        __TYPE__* xi = thrust::get<IndValAlias<INDS,N>::ind>(t); // xi = the "ind"-th argument.
        for(int k=0; k<DIM; k++) // Assume that xi and out are of size DIM, 
            out[k] = xi[k];      // and copy xi into out.
    }
    
    // Assuming that the gradient wrt. Var is GRADIN, how does it affect V ?
    // Var::DiffT<V, grad_input> = grad_input   if V == Var (in the sense that it represents the same symb. var.)
    //                             Zero(V::DIM) otherwise
    template < class V, class GRADIN >
    using DiffT = IdOrZero<Var<N,DIM,CAT>,V,GRADIN>;
};


//////////////////////////////////////////////////////////////
////             N-th PARAMETER  : Param< N >             ////
//////////////////////////////////////////////////////////////

template < int N >
struct Param
{   static const int INDEX = N; };

//////////////////////////////////////////////////////////////
////      GRADIENT OPERATOR  : Grad< F, V, Gradin >       ////
//////////////////////////////////////////////////////////////

// Computes [\partial_V F].gradin
// Symbolic differentiation is a straightforward recursive operation,
// provided that the operators have implemented their DiffT "compiler methods":
template < class F, class V, class GRADIN >
using Grad = typename F::template DiffT<V,GRADIN>; 

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
#include "formulas/constants.h"
#include "formulas/maths.h"
#include "formulas/kernels.h"
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////






