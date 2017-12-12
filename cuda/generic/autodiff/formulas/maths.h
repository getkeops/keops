/*
 * The file where the elementary math operators are defined.
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
 */



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
    INLINE void Eval(__TYPE__* params, __TYPE__* out, ARGS... args)
    {
        __TYPE__ outA[DIM], outB[DIM];
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
    INLINE void Eval(__TYPE__* params, __TYPE__* out, ARGS... args)
    {
        __TYPE__ outA[1], outB[DIM];
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
////             EXPONENTIAL : Exp< F >                   ////
//////////////////////////////////////////////////////////////

template < class F >
struct Exp
{   
    // The exponential goes from R^1 to R^1
    static const int DIM = 1;
    static_assert(F::DIM==1,"Dimension of input must be one for exp function");

    // Vars(Exp(F)) = Vars(F)
    template < int CAT >
    using VARS = typename F::VARS<CAT>;

    // To evaluate Exp(F), first evaluate F, then take its exponential...
    template < class INDS, typename... ARGS >
    INLINE void Eval(__TYPE__* params, __TYPE__* out, ARGS... args)
    {
        __TYPE__ outF[1];
        F::template Eval<INDS>(params,outF,args...);
        *out = exp(*outF);
    }
    
    // [\partial_V exp(F)].gradin = exp(F) * [\partial_V F].gradin
    template < class V, class GRADIN >
    using DiffTF = typename F::template DiffT<V,GRADIN>;
        
    template < class V, class GRADIN >
    using DiffT = Scal<Exp<F>,DiffTF<V,GRADIN>>;

};

//////////////////////////////////////////////////////////////
////             POWER OPERATOR : Pow< F, M >             ////
//////////////////////////////////////////////////////////////

template < class F, int M >
struct Pow
{
    // Pow goes from R^1 to R^1
    static const int DIM = 1;
    static_assert(F::DIM==1,"Dimension of input must be one for exp function");
    
    // Vars( F^M ) = Vars( F )
    template < int CAT >
    using VARS = typename F::VARS<CAT>;

    // To compute F^M, first compute F, then use the cmath function pow.
    template < class INDS, typename... ARGS >
    INLINE void Eval(__TYPE__* params, __TYPE__* out, ARGS... args)
    {
        __TYPE__ outF[1];
        F::template Eval<INDS>(params,outF,args...);
        *out = pow(*outF,M);
    }

    // [\partial_V F^M].gradin  =  M * (F^(M-1)) * [\partial_V F].gradin
    template < class V, class GRADIN >
    using DiffTF = typename F::template DiffT<V,GRADIN>;
        
    template < class V, class GRADIN >
    using DiffT = Scal<Scal<IntConstant<M>,Pow<F,M-1>>,DiffTF<V,GRADIN>>;

};

//////////////////////////////////////////////////////////////
////             SQUARED OPERATOR : Square< F >           ////
//////////////////////////////////////////////////////////////

template < class F >
using Square = Pow<F,2>;

//////////////////////////////////////////////////////////////
////               MINUS OPERATOR : Minus< F >            ////
//////////////////////////////////////////////////////////////

template < class F >
using Minus = Scal<IntConstant<-1>,F>;

//////////////////////////////////////////////////////////////
////               SUBTRACTION  : Subtract< A,B >         ////
//////////////////////////////////////////////////////////////

template < class FA, class FB >
using Subtract = Add<FA,Minus<FB>>;


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
struct Log {	
	static const int DIM = 1;
	static_assert(F::DIM==1,"Dimension of input must be one for exp function");

	template < int CAT >
	using VARS = typename F::VARS<CAT>;

	template < class INDS, typename... ARGS >
	INLINE void Eval(__TYPE__* params, __TYPE__* out, ARGS... args) {	
		__TYPE__ outF[1];	
		F::template Eval<INDS>(params,outF,args...);
		*out = log(*outF);		
	}

	template < class V, class GRADIN >
	using DiffTF = typename F::template DiffT<V,GRADIN>;
		
	template < class V, class GRADIN >
	using DiffT = Scal<Inv<F>,DiffTF<V,GRADIN>>;
};

//////////////////////////////////////////////////////////////
////             POWER OPERATOR : Powf< A, B >            ////
//////////////////////////////////////////////////////////////

template < class FA, class FB >
using Powf = Exp<Scal<FB,Log<FA>>>;

//////////////////////////////////////////////////////////////
////             SQUARE ROOT : Sqrt< F >                  ////
//////////////////////////////////////////////////////////////

template < class F >
using Sqrt = Powf<F,IntInv<2>>;

