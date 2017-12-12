/*
 * The file where the elementary norm-related operators are defined.
 * Available norms and scalar products are :
 * 
 *   < .,. >, | . |^2, | .-. |^2 :
 *      Scalprod<FA,FB> 			: scalar product between FA and FB
 *      SqNorm2<F>					: alias for Scalprod<F,F>
 *      SqDist<A,B>					: alias for SqNorm2<Subtract<A,B>>
 * 
 */




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
    INLINE void Eval(__TYPE__* params, __TYPE__* out, ARGS... args)
    {
        *out = 0;
        __TYPE__ outA[DIMIN], outB[DIMIN]; // Don't forget to allocate enough memory !
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
////         SQUARED L2 NORM : SqNorm2< F >               ////
//////////////////////////////////////////////////////////////

// Simple alias
template < class F >
using SqNorm2 = Scalprod<F,F>;

//////////////////////////////////////////////////////////////
////      SQUARED DISTANCE : SqDist<A,B>                  ////
//////////////////////////////////////////////////////////////

template < class X, class Y >
using SqDist = SqNorm2<Subtract<X,Y>>;




