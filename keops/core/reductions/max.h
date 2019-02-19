#pragma once

#include "core/Pack.h"

#include "core/autodiff.h"
#include "core/reductions/zero.h"

namespace keops {

// max+argmax reduction : base class

template < class F, int tagI=0 >
struct MaxArgMaxReduction_Base : public Reduction<F,tagI> {

    // We work with a (values,indices) vector
	static const int DIMRED = 2*F::DIM;	// dimension of temporary variable for reduction
		
		template < typename TYPE >
		struct InitializeReduction {
			HOST_DEVICE INLINE void operator()(TYPE *tmp) {
				for(int k=0; k<F::DIM; k++)
					tmp[k] = MINUS_INFINITY<TYPE>::value; // initialize output
				for(int k=F::DIM; k<2*F::DIM; k++)
					tmp[k] = 0; // initialize output
			}
		};

		// equivalent of the += operation
		template < typename TYPE >
		struct ReducePairShort {
			HOST_DEVICE INLINE void operator()(TYPE *tmp, TYPE *xi, int j) {
				for(int k=0; k<F::DIM; k++) {
					if(xi[k]>tmp[k]) {
						tmp[k] = xi[k];
						tmp[F::DIM+k] = j;
					}
				}
			}
		};
        
		// equivalent of the += operation
		template < typename TYPE >
		struct ReducePair {
			HOST_DEVICE INLINE void operator()(TYPE *tmp, TYPE *xi) {
				for(int k=0; k<F::DIM; k++) {
					if(xi[k]>tmp[k]) {
						tmp[k] = xi[k];
						tmp[F::DIM+k] = xi[F::DIM+k];
					}
				}
			}
		};
        
};


// Implements the max+argmax reduction operation : for each i or each j, find the maximal value of Fij anbd its index
// operation is vectorized: if Fij is vector-valued, max+argmax is computed for each dimension.

template < class F, int tagI=0 >
struct MaxArgMaxReduction : public MaxArgMaxReduction_Base<F,tagI>, UnaryOp<MaxArgMaxReduction,F,tagI> {

        static const int DIM = 2*F::DIM;		// DIM is dimension of output of convolution ; for a max-argmax reduction it is equal to 2 times the dimension of output of formula
		
    static void PrintIdString(std::stringstream& str) {
        str << "MaxArgMaxReduction";
    }
        
    template < typename TYPE >
    struct FinalizeOutput {
        HOST_DEVICE INLINE void operator()(TYPE *tmp, TYPE *out, TYPE **px, int i) {
            for(int k=0; k<DIM; k++)
                out[k] = tmp[k];
        }
    };

    // no gradient implemented here

};

// Implements the argmax reduction operation : for each i or each j, find the index of the
// maximal value of Fij
// operation is vectorized: if Fij is vector-valued, argmax is computed for each dimension.

template < class F, int tagI=0 >
struct ArgMaxReduction : public MaxArgMaxReduction_Base<F,tagI>, UnaryOp<ArgMaxReduction,F,tagI> {
        
        static const int DIM = F::DIM;		// DIM is dimension of output of convolution ; for a argmax reduction it is equal to the dimension of output of formula
		
    static void PrintIdString(std::stringstream& str) {
        str << "ArgMaxReduction";
    }

    template < typename TYPE >
    struct FinalizeOutput {
        HOST_DEVICE INLINE void operator()(TYPE *tmp, TYPE *out, TYPE **px, int i) {
            for(int k=0; k<F::DIM; k++)
                out[k] = tmp[F::DIM+k];
        }
    };

    template < class V, class GRADIN >
    using DiffT = ZeroReduction<V::DIM,(V::CAT)%2>;
    // remark : if V::CAT is 2 (parameter), we will get tagI=(V::CAT)%2=0, so we will do reduction wrt j.
    // In this case there is a summation left to be done by the user.

};

// Implements the max reduction operation : for each i or each j, find the
// maximal value of Fij
// operation is vectorized: if Fij is vector-valued, max is computed for each dimension.

template < class F, int tagI=0 >
struct MaxReduction : public MaxArgMaxReduction_Base<F,tagI>, UnaryOp<MaxReduction,F,tagI> {
        
        static const int DIM = F::DIM;		// DIM is dimension of output of convolution ; for a max reduction it is equal to the dimension of output of formula
		
    static void PrintIdString(std::stringstream& str) {
        str << "MaxReduction";
    }

    template < typename TYPE >
    struct FinalizeOutput {
        HOST_DEVICE INLINE void operator()(TYPE *tmp, TYPE *out, TYPE **px, int i) {
            for(int k=0; k<F::DIM; k++)
                out[k] = tmp[k];
        }
    };

    // no gradient implemented here

};

}
