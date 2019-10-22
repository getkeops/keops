#pragma once

#include <sstream>

#include "core/autodiff/UnaryOp.h"
#include "core/reductions/Reduction.h"
#include "core/reductions/Zero_Reduction.h"
#include "core/pre_headers.h"
#include "core/utils/Infinity.h"


namespace keops {

// max+argmax reduction : base class

template < class F, int tagI=0 >
struct Max_ArgMax_Reduction_Base : public Reduction<F,tagI> {

    // We work with a (values,indices) vector
	static const int DIMRED = 2*F::DIM;	// dimension of temporary variable for reduction
		
		template < typename TYPE >
		struct InitializeReduction {
			DEVICE INLINE void operator()(TYPE *tmp) {
#pragma unroll
				for(int k=0; k<F::DIM; k++)
					tmp[k] = NEG_INFINITY<TYPE>::value; // initialize output
#pragma unroll
				for(int k=F::DIM; k<2*F::DIM; k++)
					tmp[k] = 0; // initialize output
			}
		};

		// equivalent of the += operation
		template < typename TYPEACC, typename TYPE >
		struct ReducePairShort {
			DEVICE INLINE void operator()(TYPEACC *tmp, TYPE *xi, int j) {
#pragma unroll
				for(int k=0; k<F::DIM; k++) {
					if(xi[k]>tmp[k]) {
						tmp[k] = xi[k];
						tmp[F::DIM+k] = j;
					}
				}
			}
		};
        
		// equivalent of the += operation
		template < typename TYPEACC, typename TYPE >
		struct ReducePair {
			DEVICE INLINE void operator()(TYPEACC *tmp, TYPE *xi) {
#pragma unroll
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
struct Max_ArgMax_Reduction : public Max_ArgMax_Reduction_Base<F,tagI>, UnaryOp<Max_ArgMax_Reduction,F,tagI> {

        static const int DIM = 2*F::DIM;		// DIM is dimension of output of convolution ; for a max-argmax reduction it is equal to 2 times the dimension of output of formula
		
    static void PrintIdString(::std::stringstream& str) {
        str << "Max_ArgMax_Reduction";
    }
        
    template < typename TYPEACC, typename TYPE >
    struct FinalizeOutput {
        DEVICE INLINE void operator()(TYPEACC *acc, TYPE *out, TYPE **px, int i) {
#pragma unroll
            for(int k=0; k<DIM; k++)
                out[k] = acc[k];
        }
    };

    // no gradient implemented here

};

// Implements the argmax reduction operation : for each i or each j, find the index of the
// maximal value of Fij
// operation is vectorized: if Fij is vector-valued, argmax is computed for each dimension.

template < class F, int tagI=0 >
struct ArgMax_Reduction : public Max_ArgMax_Reduction_Base<F,tagI>, UnaryOp<ArgMax_Reduction,F,tagI> {
        
        static const int DIM = F::DIM;		// DIM is dimension of output of convolution ; for a argmax reduction it is equal to the dimension of output of formula
		
    static void PrintIdString(::std::stringstream& str) {
        str << "ArgMax_Reduction";
    }

    template < typename TYPEACC, typename TYPE >
    struct FinalizeOutput {
        DEVICE INLINE void operator()(TYPEACC *acc, TYPE *out, TYPE **px, int i) {
#pragma unroll
            for(int k=0; k<F::DIM; k++)
                out[k] = acc[F::DIM+k];
        }
    };

    template < class V, class GRADIN >
    using DiffT = Zero_Reduction<V::DIM,(V::CAT)%2>;
    // remark : if V::CAT is 2 (parameter), we will get tagI=(V::CAT)%2=0, so we will do reduction wrt j.
    // In this case there is a summation left to be done by the user.

};

// Implements the max reduction operation : for each i or each j, find the
// maximal value of Fij
// operation is vectorized: if Fij is vector-valued, max is computed for each dimension.

template < class F, int tagI=0 >
struct Max_Reduction : public Max_ArgMax_Reduction_Base<F,tagI>, UnaryOp<Max_Reduction,F,tagI> {
        
        static const int DIM = F::DIM;		// DIM is dimension of output of convolution ; for a max reduction it is equal to the dimension of output of formula
		
    static void PrintIdString(::std::stringstream& str) {
        str << "Max_Reduction";
    }

    template < typename TYPEACC, typename TYPE >
    struct FinalizeOutput {
        DEVICE INLINE void operator()(TYPEACC *acc, TYPE *out, TYPE **px, int i) {
#pragma unroll
            for(int k=0; k<F::DIM; k++)
                out[k] = acc[k];
        }
    };

    // no gradient implemented here

};

#define ArgMax_Reduction(F,I) KeopsNS<ArgMax_Reduction<decltype(InvKeopsNS(F)),I>>()
#define Max_Reduction(F,I) KeopsNS<Max_Reduction<decltype(InvKeopsNS(F)),I>>()
#define Max_ArgMax_Reduction(F,I) KeopsNS<Max_ArgMax_Reduction<decltype(InvKeopsNS(F)),I>>()


}
