#pragma once

#include <assert.h>
#include <sstream>

#include "core/pack/CondType.h"
#include "core/autodiff/UnaryOp.h"
#include "core/autodiff/BinaryOp.h"
#include "core/formulas/maths/Add.h"
#include "core/formulas/norms/SqNormDiag.h"
#include "core/formulas/norms/SqNormIso.h"
#include "core/pre_headers.h"


namespace keops {

///////////////////////////////////////////////////////////////////////////////////////
////             Fully anisotropic norm, if S::DIM == A::DIM * A::DIM:             ////
///////////////////////////////////////////////////////////////////////////////////////

template < class A, class X > struct SymTwoDot;

// TODO: The SymTwoOuterProduct, SymTwoDot, SymOuterProduct methods should be
//       implemented with the generic methods Tensordot or with MatVecMult, TensorProd, etc...

// SymTwoOuterProduct<X,Y> = X @ Y^T + Y @ X^T
template < class X, class Y >
struct SymTwoOuterProduct : BinaryOp<SymTwoOuterProduct,X,Y> {
  // Output dimension = X::DIM**2, provided that X::DIM == Y::DIM
  static const int DIMIN = X::DIM;
  static_assert( Y::DIM == DIMIN, "A symmetric outer product can only be done with two vectors sharing the same length.");
  static const int DIM = DIMIN * DIMIN;

  static void PrintIdString(::std::stringstream& str) {
    str << "<SymTwoOuterProduct>";
  }

  static DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outX, __TYPE__ *outY) {
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

  static void PrintIdString(::std::stringstream& str) {
    str << "<SymTwoDot>";
  }

  static DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outA, __TYPE__ *outX) {
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

  static void PrintIdString(::std::stringstream& str) {
    str << "SymOuterProduct";
  }

  static DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outX) {
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

  static void PrintIdString(::std::stringstream& str) {
    str << "<SymSqNorm>";
  }

  static DEVICE INLINE void Operation(__TYPE__ *out, __TYPE__ *outA, __TYPE__ *outX) {
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
  Add< typename A::template DiffT<V, SymOuterProduct< X > >,
       typename X::template DiffT<V, SymTwoDot<    A, X > >   > >;

};

template < class A, class X >
using WeightedSqNorm = CondType< SqNormIso<A,X>,
CondType< SqNormDiag<A,X>, SymSqNorm<A,X>, A::DIM==X::DIM >,
A::DIM == 1  >;

#define WeightedSqNorm(s,f)   KeopsNS<WeightedSqNorm<decltype(InvKeopsNS(s)), decltype(InvKeopsNS(f))>>()



}
