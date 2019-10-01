#pragma once

#include <sstream>

#include "core/formulas/maths/Extract.h"
#include "core/reductions/Reduction.h"
#include "core/reductions/Sum_Reduction.h"
#include "core/formulas/maths/Concat.h"
#include "core/pre_headers.h"


namespace keops {

// N.B.: additional arguments D and G_ have to be in this order to fit the
//       conventions of the Genred binder.
template < class F, int D, int tagI = 0, class G_ = IntConstant< 0 > > 
struct Scattered_Sum_Reduction : public Reduction< Concat< G_, F >, tagI > {

  using G = G_;

  using PARENT = Reduction< Concat< G_, F >, tagI >;

  // dimension of temporary variable for reduction:
  static const int DIMRED = D * F::DIM;  // D concatenated copies of F(..)

  static const int DIM = DIMRED;  // Dimension of the output

  static_assert(G::DIM == 1, "Scattered Sum requires a scalar index of dimension 1.");

  static void PrintId(::std::stringstream &str) {
    str << "Scattered_Sum_Reduction(F=";            // prints "("
    F::PrintId(str);   // prints the formula F
    str << ",G=";
    G::PrintId(str);   // prints the formula G
    str << ",D=" << D << ",tagI=" << tagI << ")";
  }

  template < typename TYPE >
  struct InitializeReduction {
    DEVICE INLINE void operator()(TYPE *tmp) {
      // We fill empty cells with the neutral element of the reduction operation 0
      for (int k = 0; k < DIMRED; k++)
        tmp[k] = 0.0f;
    }
  };

  // equivalent of the += operation
  template < typename TYPE >
  struct ReducePairShort {
    DEVICE INLINE void operator()(TYPE *tmp, TYPE *xi, int j) {
      // The formula is Concat(G,F) -->  xi = [G(..),    F(..)  ]
      //                                       dim=1, dim=F::DIM
      int index = round(xi[0]);  // read the value of "G"

      // Hopefully, the compiler will handle the branching efficiently...
      if (0 <= index && index < D) {  // Boundary conditions
        for (int k = 0; k < F::DIM; k++) {
          tmp[index * F::DIM + k] += xi[1 + k];  // Increment the buffer at position "index" with the value of F
        }
      }
    }
  };

  // equivalent of the += operation
  template < typename TYPE >
  struct ReducePair {
    DEVICE INLINE void operator()(TYPE *tmp, TYPE *xi) {
      // The formula is Concat(G,F) -->  xi = [G(..),    F(..)  ]
      //                                       dim=1, dim=F::DIM
      int index = round(xi[0]);  // read the value of "G"

      // Hopefully, the compiler will handle the branching efficiently...
      if (0 <= index && index < D) {  // Boundary conditions. 
        for (int k = 0; k < F::DIM; k++) {
          tmp[index * F::DIM + k] += xi[1 + k];  // Increment the buffer at position "index" with the value of F
        }
      }
    }
  };

  template < typename TYPE >
  struct FinalizeOutput {  // Copy the buffer tmp (register) onto the output (device memory)
    DEVICE INLINE void operator()(TYPE *tmp, TYPE *out, TYPE **px, int i) {
      for (int k = 0; k < DIM; k++)
        out[k] = tmp[k];
    }
  };


};

#define Scattered_Sum_Reduction(F, D, I, G) KeopsNS<Scattered_Sum_Reduction<decltype(InvKeopsNS(F)),D,I,decltype(InvKeopsNS(G))>>()

}
