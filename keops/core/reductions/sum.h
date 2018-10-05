#pragma once

#include "core/Pack.h"

#include "core/autodiff.h"

#include "core/reductions/reduction.h"

namespace keops {

// Implements the summation reduction operation

template < class F, int tagI > struct SumReductionAlias;

template < class F, int tagI=0 > 
using SumReduction = typename SumReductionAlias<F,tagI>::type; 

template < class F, int tagI >
struct SumReductionImpl : public Reduction<F,tagI> {

    static const int DIM = F::DIM;		// DIM is dimension of output of convolution ; for a sum reduction it is equal to the dimension of output of formula

    static const int DIMRED = DIM;		// dimension of temporary variable for reduction

	// recursive function to print the formula as a string 
    static void PrintId(std::stringstream& str) {
        str << "SumReduction (with tagI=" << tagI << ") of :" << std::endl;
        str << PrintFormula<F>();				// prints the formula F
    }

    template < typename TYPE >
    struct InitializeReduction {
        HOST_DEVICE INLINE void operator()(TYPE *tmp) {
            for(int k=0; k<DIM; k++)
                tmp[k] = 0.0f; // initialize output
        }
    };

    // equivalent of the += operation
    template < typename TYPE >
    struct ReducePairShort {
        HOST_DEVICE INLINE void operator()(TYPE *tmp, TYPE *xi, int j) {
            for(int k=0; k<DIM; k++) {
                tmp[k] += xi[k];
            }
        }
    };

    // equivalent of the += operation
    template < typename TYPE >
    struct ReducePair {
        HOST_DEVICE INLINE void operator()(TYPE *tmp, TYPE *xi) {
            for(int k=0; k<DIM; k++) {
                tmp[k] += xi[k];
            }
        }
    };

    template < typename TYPE >
    struct FinalizeOutput {
        HOST_DEVICE INLINE void operator()(TYPE *tmp, TYPE *out, TYPE **px, int i) {
            for(int k=0; k<DIM; k++)
                out[k] = tmp[k];
        }
    };

    template < class V, class GRADIN, class FO=void >
    using DiffT = SumReduction<Grad<F,V,GRADIN>,(V::CAT)%2>;
    // remark : if V::CAT is 2 (parameter), we will get tagI=(V::CAT)%2=0, so we will do reduction wrt j.
    // In this case there is a summation left to be done by the user.

};

template < class F, int tagI >
struct SumReductionAlias {
    using type = SumReductionImpl<F,tagI>;
};

template < int DIM, int tagI > struct ZeroReduction;

template < int DIM, int tagI >
struct SumReductionAlias<Zero<DIM>,tagI> {
    using type = ZeroReduction<DIM,tagI>;
};


}
