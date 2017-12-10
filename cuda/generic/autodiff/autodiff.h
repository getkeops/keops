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
//////////////////////////////////////////////////////////////

/* Class for base variable
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
    INLINE void Eval(float* params, float* out, ARGS... args)
    {
        auto t = thrust::make_tuple(args...); // let us access the args using indexing syntax
        // IndValAlias<INDS,N>::ind is the first index such that INDS[ind]==N. Let's call it "ind"
        float* xi = thrust::get<IndValAlias<INDS,N>::ind>(t); // xi = the "ind"-th argument.
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
//////////////////////////////////////////////////////////////

// A "zero" vector of size _DIM
// Declared using the   Zero<DIM>   syntax.
template < int _DIM >
struct Zero
{
    static const int DIM = _DIM;

    template < int CAT >      // Whatever CAT... 
    using VARS = univpack<>;  // there's no variable used in there.

    // Evaluation is easy : simply fill-up *out with zeros.
    template < class INDS, typename... ARGS >
    INLINE void Eval(float* params, float* out, ARGS... args)
    {
        for(int k=0; k<DIM; k++)
            out[k] = 0;
    }
    
    // There is no gradient to accumulate on V, whatever V.    
    template < class V, class GRADIN >
    using DiffT = Zero<V::DIM>;
};

// A constant integer value, defined using the     IntConstant<N>    syntax.
template < int N >
struct IntConstant
{
    static const int DIM = 1;

    template < int CAT >      // Whatever CAT... 
    using VARS = univpack<>;  // there's no variable used in there.
    
    // Evaluation is easy : simply fill *out = out[0] with N.
    template < class INDS, typename... ARGS >
    INLINE void Eval(float* params, float* out, ARGS... args)
    {
        *out = N;
    }
        
    // There is no gradient to accumulate on V, whatever V.
    template < class V, class GRADIN >
    using DiffT = Zero<V::DIM>;
};


// A constant parameter value, a scalar (but we may use a pointer ?)
template < class PRM >
struct Constant
{
    static const int DIM = 1; // Scalar-valued parameters only.

    // A parameter is a variable of category "2" ( 0 = Xi, 1 = Yj )
    template < int CAT >
    using VARS = CondType<univpack<PRM>,univpack<>,CAT==2>;

    // "returns" the appropriate value in the params array.
    template < class INDS, typename... ARGS >
    INLINE void Eval(float* params, float* out, ARGS... args)
    {
        *out = params[PRM::INDEX];
    }
    
    // There's no gradient to accumulate in V, whatever V.    
    template < class V, class GRADIN >
    using DiffT = Zero<V::DIM>;
};


//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

// Addition, Scalar product and "Scalar*Vector product" symbolic operators.
// The actual implementation can be found below.
// Since the gradients of these operations are "bootstrapped", we need to be a little bit
// careful with the declaration order, and therefore use three "typenames" per operation:
// OpAlias, OpImpl and Op (proper).
template < class FA, class FB > struct AddAlias;
template < class FA, class FB > struct ScalprodAlias;
template < class FA, class FB > struct ScalAlias;
template < class F > struct Norm2Alias;

template < class FA, class FB >
using Add = typename AddAlias<FA,FB>::type;

template < class FA, class FB >
using Scalprod = typename ScalprodAlias<FA,FB>::type;

template < class FA, class FB >
using Scal = typename ScalAlias<FA,FB>::type;


//////////////////////////////////////////////////////////////
////               ADDITION : Add< FA,FB >                ////
//////////////////////////////////////////////////////////////


template < class FA, class FB >
struct AddImpl
{
    // Output dim = FA::DIM = FB::DIM
    static const int DIM = FA::DIM;
    static_assert(DIM==FB::DIM,"Dimensions must be the same for Add");
    
    // Vars( A + B ) = Vars(A) U Vars(B), whatever the category
    template < int CAT >
    using VARS = MergePacks<typename FA::VARS<CAT>,typename FB::VARS<CAT>>;

    // To evaluate A + B, first evaluate A, then B, and then add the result and put it in "out".
    template < class INDS, typename... ARGS >
    INLINE void Eval(float* params, float* out, ARGS... args)
    {
        float outA[DIM], outB[DIM];
        FA::template Eval<INDS>(params,outA,args...);
        FB::template Eval<INDS>(params,outB,args...);
        for(int k=0; k<DIM; k++)
            out[k] = outA[k] + outB[k];
    }

    // [\partial_V (A + B) ] . gradin = [\partial_V A ] . gradin  + [\partial_V B ] . gradin 
    template < class V, class GRADIN >
    using DiffTA = typename FA::template DiffT<V,GRADIN>;
    
    template < class V, class GRADIN >
    using DiffTB = typename FB::template DiffT<V,GRADIN>;
    
    template < class V, class GRADIN >
    using DiffT = Add < DiffTA<V,GRADIN> , DiffTB<V,GRADIN> >;

};

template < class FA, class FB >
struct AddAlias
{   using type = AddImpl<FA,FB>; };

// Constants, etc. will lead to the creation of *many* zero vectors when computing the gradient.
// Even though this backpropagation engine makes few optimizations, 
// this is definitely the one that should not be forgotten.

// A + 0 = A
template < class FA, int DIM >
struct AddAlias<FA,Zero<DIM>>
{
    static_assert(DIM==FA::DIM,"Dimensions must be the same for Add");
    using type = FA;
};

// 0 + B = B
template < class FB, int DIM >
struct AddAlias<Zero<DIM>,FB>
{
    static_assert(DIM==FB::DIM,"Dimensions must be the same for Add");
    using type = FB;
};

// 0 + 0 = la tete a Toto
template < int DIM1, int DIM2 >
struct AddAlias<Zero<DIM1>,Zero<DIM2>>
{
    static_assert(DIM1==DIM2,"Dimensions must be the same for Add");
    using type = Zero<DIM1>;
};


//////////////////////////////////////////////////////////////
////      Scal*Vector Multiplication : Scal< FA,FB>       ////
//////////////////////////////////////////////////////////////


template < class FA, class FB >
struct ScalImpl
{
    // FB is a vector, Output has the same size, and FA is a scalar
    static const int DIM = FB::DIM;
    static_assert(FA::DIM==1,"Dimension of FA must be 1 for Scal");

    // Vars( A * B ) = Vars(A) U Vars(B)
    template < int CAT >
    using VARS = MergePacks<typename FA::VARS<CAT>,typename FB::VARS<CAT>>;

    // To evaluate A*B, first evaluate A, then B, then store the pointwise mult. in out.
    template < class INDS, typename... ARGS >
    INLINE void Eval(float* params, float* out, ARGS... args)
    {
        float outA[1], outB[DIM];
        FA::template Eval<INDS>(params,outA,args...);
        FB::template Eval<INDS>(params,outB,args...);
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
    using DiffTA = typename FA::template DiffT<V,GRADIN>;
    
    template < class V, class GRADIN >
    using DiffTB = typename FB::template DiffT<V,GRADIN>;
    
    template < class V, class GRADIN >
    using DiffT = Add < DiffTA<V,Scalprod<GRADIN,FB>> , Scal < FA, DiffTB<V,GRADIN> > >;

};


template < class FA, class FB >
struct ScalAlias
{
    using type = ScalImpl<FA,FB>;
};


// Constants, etc. will lead to the creation of *many* zero vectors when computing the gradient.
// Even though this backpropagation engine makes few optimizations, 
// this is definitely the one that should not be forgotten.

// A * 0 = 0
template < class FA, int DIM >
struct ScalAlias<FA,Zero<DIM>>
{
    static_assert(1==FA::DIM,"Dimension of FA must be 1 for Scal");
    using type = Zero<DIM>;
};

// 0 * B = 0
template < class FB, int DIM >
struct ScalAlias<Zero<DIM>,FB>
{
    static_assert(DIM==1,"Dimension of FA must be 1 for Scal");
    using type = Zero<FB::DIM>;
};

// 0 * 0 = 0
template < int DIM1, int DIM2 >
struct ScalAlias<Zero<DIM1>,Zero<DIM2>>
{
    static_assert(DIM1==1,"Dimension of FA must be 1 for Scal");
    using type = Zero<DIM2>;
};


//////////////////////////////////////////////////////////////
////           SCALAR PRODUCT :   Scalprod< A,B >         ////
//////////////////////////////////////////////////////////////



template < class FA, class FB >
struct ScalprodImpl
{
    // Output dimension = 1, provided that FA::DIM = FB::DIM
    static const int DIMIN = FA::DIM;
    static_assert(DIMIN==FB::DIM,"Dimensions must be the same for Scalprod");
    static const int DIM = 1;

    // Vars(<A,B>) = Vars(A) U Vars(B)
    template < int CAT >
    using VARS = MergePacks<typename FA::VARS<CAT>,typename FB::VARS<CAT>>;
    
    // To evaluate the scalar <A,B>, first evaluate A, then B, then proceed to the summation.
    template < class INDS, typename... ARGS >
    INLINE void Eval(float* params, float* out, ARGS... args)
    {
        *out = 0;
        float outA[DIMIN], outB[DIMIN]; // Don't forget to allocate enough memory !
        FA::template Eval<INDS>(params,outA,args...);
        FB::template Eval<INDS>(params,outB,args...);
        for(int k=0; k<DIMIN; k++)
            *out += outA[k]*outB[k];
    }
    
    // <A,B> is scalar-valued, so that gradin is necessarily a scalar.
    // [\partial_V <A,B>].gradin = gradin * ( [\partial_V A].B + [\partial_V B].A )
    template < class V, class GRADIN >
    using DiffTA = typename FA::template DiffT<V,GRADIN>;
    
    template < class V, class GRADIN >
    using DiffTB = typename FB::template DiffT<V,GRADIN>;
    
    template < class V, class GRADIN >
    using DiffT = Scal < GRADIN , Add < DiffTA<V,FB> , DiffTB<V,FA> > >;
};


template < class FA, class FB >
struct ScalprodAlias
{
    using type = ScalprodImpl<FA,FB>;
};

// Three simple optimizations :

// <A,0> = 0
template < class FA, int DIM >
struct ScalprodAlias<FA,Zero<DIM>>
{
    static_assert(DIM==FA::DIM,"Dimensions must be the same for Scalprod");
    using type = Zero<1>;
};

// <0,B> = 0
template < class FB, int DIM >
struct ScalprodAlias<Zero<DIM>,FB>
{
    static_assert(DIM==FB::DIM,"Dimensions must be the same for Scalprod");
    using type = Zero<1>;
};

// <0,0> = 0
template < int DIM1, int DIM2 >
struct ScalprodAlias<Zero<DIM1>,Zero<DIM2>>
{
    static_assert(DIM1==DIM2,"Dimensions must be the same for Scalprod");
    using type = Zero<1>;
};


//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////


template < class F >
using SqNorm2 = Scalprod<F,F>;

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////



template < class F >
struct Exp
{	
    static const int DIM = 1;
    static_assert(F::DIM==1,"Dimension of input must be one for exp function");

    template < int CAT >
    using VARS = typename F::VARS<CAT>;

    template < class INDS, typename... ARGS >
    INLINE void Eval(float* params, float* out, ARGS... args)
    {	
        float outF[1];	
        F::template Eval<INDS>(params,outF,args...);
        *out = exp(*outF);		
    }

    template < class V, class GRADIN >
    using DiffTF = typename F::template DiffT<V,GRADIN>;
        
    template < class V, class GRADIN >
    using DiffT = Scal<Exp<F>,DiffTF<V,GRADIN>>;

};


template < class F, int M >
struct Pow
{	
    static const int DIM = 1;
    static_assert(F::DIM==1,"Dimension of input must be one for exp function");

    template < int CAT >
    using VARS = typename F::VARS<CAT>;

    template < class INDS, typename... ARGS >
    INLINE void Eval(float* params, float* out, ARGS... args)
    {	
        float outF[1];	
        F::template Eval<INDS>(params,outF,args...);
        *out = pow(*outF,M);		
    }

    template < class V, class GRADIN >
    using DiffTF = typename F::template DiffT<V,GRADIN>;
        
    template < class V, class GRADIN >
    using DiffT = Scal<Scal<IntConstant<M>,Pow<F,M-1>>,DiffTF<V,GRADIN>>;

};

template < class F >
using Square = Pow<F,2>;


template < class F >
using Minus = Scal<IntConstant<-1>,F>;

template < class FA, class FB >
using Subtract = Add<FA,Minus<FB>>;



template < class OOS2, class X, class Y, class Beta >
using GaussKernel = Scal<Exp<Scal<Constant<OOS2>,Minus<SqNorm2<Subtract<X,Y>>>>>,Beta>;




template < int N >
struct Param
{
    static const int INDEX = N;
};


template < class F, class V, class GRADIN >
using Grad = typename F::template DiffT<V,GRADIN>;

