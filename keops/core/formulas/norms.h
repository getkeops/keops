#pragma once

#include <sstream>
#include <assert.h>

#include "core/Pack.h"
#include "core/autodiff.h"

#include "core/formulas/constants.h"
#include "core/formulas/maths.h"

/*
 * The file where the elementary norm-related operators are defined.
 * Available norms and scalar products are :
 *
 *   (.|.), |.|, |.|^2, |.-.|^2 :
 *      Scalprod<FA,FB> 			: scalar product between FA and FB
 *      SqNorm2<F>					: alias for Scalprod<F,F>
 *      Norm2<F>					: alias for Sqrt<SqNorm2<F>>
 *      SqDist<FA,FB>				: alias for SqNorm2<Subtract<FA,FB>>
 *   Non-standard norms :
 *      WeightedSqNorm<A,F>         : squared weighted norm of F, either :
 *                                       - a * sum_k f_k^2 if A::DIM=1
 *                                       - sum_k a_k f_k^2 if A::DIM=F::DIM
 *                                       - sum_kl a_kl f_k f_l if A::DIM=F::DIM^2
 *      WeightedSqDist<A,FA,FB>       : alias for WeightedSqNorm<A,Subtract<FA,FB>>
 *
 */




//////////////////////////////////////////////////////////////
////           SCALAR PRODUCT :   Scalprod< A,B >         ////
//////////////////////////////////////////////////////////////

namespace keops {

template < class FA, class FB >
struct Scalprod_Impl : BinaryOp<Scalprod_Impl,FA,FB> {
    // Output dimension = 1, provided that FA::DIM = FB::DIM
    static const int DIMIN = FA::DIM;
    static_assert(DIMIN==FB::DIM,"Dimensions must be the same for Scalprod");
    static const int DIM = 1;

    static void PrintIdString(std::stringstream& str) { str << "|"; }
    
    static HOST_DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outA, __TYPE__ *outB) {
    		*out = 0;
            for(int k=0; k<DIMIN; k++)
            	*out += outA[k]*outB[k];
	}

    // <A,B> is scalar-valued, so that gradin is necessarily a scalar.
    // [\partial_V <A,B>].gradin = gradin * ( [\partial_V A].B + [\partial_V B].A )
    template < class V, class GRADIN >
    using DiffT = Scal < GRADIN , Add < typename FA::template DiffT<V,FB> , typename FB::template DiffT<V,FA> > >;
};


template < class FA, class FB >
struct Scalprod_Alias {
    using type = Scalprod_Impl<FA,FB>;
};

// Three simple optimizations :

// <A,0> = 0
template < class FA, int DIM >
struct Scalprod_Alias<FA,Zero<DIM>> {
    static_assert(DIM==FA::DIM,"Dimensions must be the same for Scalprod");
    using type = Zero<1>;
};

// <0,B> = 0
template < class FB, int DIM >
struct Scalprod_Alias<Zero<DIM>,FB> {
    static_assert(DIM==FB::DIM,"Dimensions must be the same for Scalprod");
    using type = Zero<1>;
};

// <0,0> = 0
template < int DIM1, int DIM2 >
struct Scalprod_Alias<Zero<DIM1>,Zero<DIM2>> {
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
////           ANISOTROPIC NORM :   SqNorm< S,A >         ////
//////////////////////////////////////////////////////////////

// Isotropic norm, if S is a scalar:
// SqNormIso<S,A> = S * <A,A> = S * sum_i a_i*a_i
template < class FS, class FA >
struct SqNormIso : BinaryOp<SqNormIso,FS,FA> {
    // Output dimension = 1, provided that FS::DIM = 1
    static const int DIMIN = FA::DIM;
    static_assert(FS::DIM==1,"Isotropic square norm expects a scalar parameter.");
    static const int DIM = 1;

    static void PrintIdString(std::stringstream& str) { str << "<SqNormIso>"; }
    
    static HOST_DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outS, __TYPE__ *outA) {
    		*out = 0;
            for(int k=0; k<DIMIN; k++)
            	*out += outA[k]*outA[k];
            *out *= *outS;
	}

    // S*<A,A> is scalar-valued, so that gradin is necessarily a scalar.
    // [\partial_V S*<A,A>].gradin = gradin * ( 2*S*[\partial_V A].A + [\partial_V S].<A,A> )
    template < class V, class GRADIN >
    using DiffT = Scal < GRADIN , 
                         Add < Scal< Scal<IntConstant<2>,FS>, typename FA::template DiffT<V,FA> >, 
                               typename FS::template DiffT<V, SqNorm2<FA> > 
                             > 
                        >;
};


// Anisotropic (but diagonal) norm, if S::DIM == A::DIM:
// SqNormDiag<S,A> = sum_i s_i*a_i*a_i
template < class FS, class FA >
struct SqNormDiag : BinaryOp<SqNormDiag,FS,FA> {
    // Output dimension = 1, provided that FS::DIM = FA::DIM
    static const int DIMIN = FA::DIM;
    static_assert(FS::DIM==FA::DIM,"Diagonal square norm expects a vector of parameters of dimension FA::DIM.");
    static const int DIM = 1;

    static void PrintIdString(std::stringstream& str) { str << "<SqNormDiag>"; }
    
    static HOST_DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outS, __TYPE__ *outA) {
    		*out = 0;
            for(int k=0; k<DIMIN; k++)
            	*out += outS[k]*outA[k]*outA[k];
	}

    // sum_i s_i*a_i*a_i is scalar-valued, so that gradin is necessarily a scalar.
    // [\partial_V ...].gradin = gradin * ( 2*[\partial_V A].(S*A) + [\partial_V S].(A*A) )
    template < class V, class GRADIN >
    using DiffT = Scal < GRADIN , 
                         Add < Scal< IntConstant<2>, typename FA::template DiffT<V,Mult<FS,FA>> >, 
                               typename FS::template DiffT<V, Mult<FA,FA> > 
                             > 
                        >;
};

// ------------------------------------------------------------------------------
// Fully anisotropic norm, if S::DIM == A::DIM * A::DIM:
// ------------------------------------------------------------------------------
template < class A, class X > struct SymTwoDot;

// SymTwoOuterProduct<X,Y> = X @ Y^T + Y @ X^T
template < class X, class Y >
struct SymTwoOuterProduct : BinaryOp<SymTwoOuterProduct,X,Y> {
    // Output dimension = X::DIM**2, provided that X::DIM == Y::DIM
    static const int DIMIN = X::DIM;
    static_assert( Y::DIM == DIMIN, "A symmetric outer product can only be done with two vectors sharing the same length.");
    static const int DIM = DIMIN * DIMIN;

    static void PrintIdString(std::stringstream& str) { str << "<SymTwoOuterProduct>"; }
    
    static HOST_DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outX, __TYPE__ *outY) {
            for(int k=0; k < DIMIN; k++) {
                for(int l=0; l < DIMIN; l++)
                    out[ k*DIMIN + l ] = outX[k] * outY[l] + outX[l] * outY[k] ;
            }
	}

    // [\partial_V (X @ Y^T + Y @ X^T)].A = [\partial_V X].(2*A@Y) + [\partial_V Y].(2*A@X)
    template < class V, class GRADIN >
    using DiffT = Add< typename X::template DiffT<V, SymTwoDot< GRADIN, Y > >,
                       typename Y::template DiffT<V, SymTwoDot< GRADIN, X > > 
                     >;
};


// SymTwoDot<A,X> = 2 * A@X (matrix product)
template < class A, class X >
struct SymTwoDot : BinaryOp<SymTwoDot,A,X> {
    // Output dimension = X::DIM, provided that A::DIM = (X::DIM)**2
    static const int DIMIN = X::DIM;
    static_assert( A::DIM == DIMIN*DIMIN, "A symmetric matrix on a space of dim D should be encoded as a vector of size D*D.");
    static const int DIM = DIMIN;

    static void PrintIdString(std::stringstream& str) { str << "<SymTwoDot>"; }
    
    static HOST_DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outA, __TYPE__ *outX) {
            for(int k=0; k < DIMIN; k++) {
                out[k] = 0;
                for(int l=0; l < DIMIN; l++) {
                    out[ k ] += outA[ k*DIMIN + l ] * outX[ l ];
                }
                out[k] *= 2;
            }
	}

    // ASSUMING THAT "A" IS A SYMMETRIC MATRIX,
    // [\partial_V 2A@X].gradin = [\partial_V X].(2*A@gradin) + [\partial_V A].(X @ gradin^T + gradin^T @ X)
    template < class V, class GRADIN >
    using DiffT = Add< typename X::template DiffT<V, SymTwoDot< A, GRADIN > >,
                       typename A::template DiffT<V, SymTwoOuterProduct< X, GRADIN > > 
                     >;
};

// SymOuterProduct<X> = X * X^T
template < class X >
struct SymOuterProduct : UnaryOp<SymOuterProduct,X> {
    // Output dimension = X::DIM**2
    static const int DIMIN = X::DIM;
    static const int DIM = DIMIN * DIMIN;

    static void PrintIdString(std::stringstream& str) { str << "SymOuterProduct"; }
    
    static HOST_DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outX) {
            for(int k=0; k < DIMIN; k++) {
                for(int l=0; l < DIMIN; l++)
                    out[ k*DIMIN + l ] = outX[ k ] * outX[ l ];
            }
	}

    // ASSUMING THAT "A" IS A SYMMETRIC MATRIX,
    // [\partial_V X*X^T].A = [\partial_V X].(2*A@X)
    template < class V, class GRADIN >
    using DiffT = typename X::template DiffT<V, SymTwoDot< GRADIN, X > >;
};




// SymSqNorm<A,X> = sum_{ij} a_ij * x_i*x_j
template < class A, class X >
struct SymSqNorm : BinaryOp<SymSqNorm,A,X> {
    // Output dimension = 1, provided that A::DIM = X::DIM**2
    static const int DIMIN = X::DIM;
    static_assert( A::DIM == X::DIM * X::DIM, "Anisotropic square norm expects a vector of parameters of dimension FA::DIM * FA::DIM.");
    static const int DIM = 1;

    static void PrintIdString(std::stringstream& str) { str << "<SymSqNorm>"; }
    
    static HOST_DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outA, __TYPE__ *outX) {
    		*out = 0;
            for(int k=0; k < DIMIN; k++) {
                for(int l=0; l < DIMIN; l++)
                    *out += outA[ k*DIMIN + l ] * outX[k]*outX[l];
            }
	}

    // ASSUMING THAT "A" IS A SYMMETRIC MATRIX,
    // sum_ij a_ij*x_i*x_j is scalar-valued, so that gradin is necessarily a scalar.
    // [\partial_V X^T @ A @ X].gradin = gradin * ( [\partial_V A].(X @ X^T) + [\partial_V X].(2*A@X) )
    template < class V, class GRADIN >
    using DiffT = Scal < GRADIN,
                        Add< typename A::template DiffT<V, SymOuterProduct< X > > ,
                             typename X::template DiffT<V, SymTwoDot<    A, X > >   > >;

};

template < class A, class X >
using WeightedSqNorm = CondType< SqNormIso<A,X> ,  
                                 CondType< SqNormDiag<A,X>, SymSqNorm<A,X>, A::DIM==X::DIM > , 
                                 A::DIM == 1  >;



//////////////////////////////////////////////////////////////
////           L2 NORM :   ||F||                          ////
//////////////////////////////////////////////////////////////

// Simple alias
template < class F >
using Norm2 = Sqrt<Scalprod<F,F>>;


//////////////////////////////////////////////////////////////
////       NORMALIZE :   F / ||F||                        ////
//////////////////////////////////////////////////////////////

// Simple alias
template < class F >
using Normalize = Scal<Rsqrt<SqNorm2<F>>,F>;


//////////////////////////////////////////////////////////////
////      SQUARED DISTANCE : SqDist<A,B>                  ////
//////////////////////////////////////////////////////////////

template < class X, class Y >
using SqDist = SqNorm2<Subtract<X,Y>>;


//////////////////////////////////////////////////////////////
////   WEIGHTED SQUARED DISTANCE : WeightedSqDist<S,A,B>  ////
//////////////////////////////////////////////////////////////

template < class S, class X, class Y >
using WeightedSqDist = WeightedSqNorm< S, Subtract<X,Y>>;

}
