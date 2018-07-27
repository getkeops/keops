#pragma once

#include "core/Pack.h"

#include "core/autodiff.h"

namespace keops {
// Implements the min+argmin reduction operation : for each i or each j, find the minimal value of Fij anbd its index
// operation is vectorized: if Fij is vector-valued, min+argmin is computed for each dimension.
// tagI is equal:
// - to 0 if you do the reduction over j (with i the index of the output vector),
// - to 1 if you do the reduction over i (with j the index of the output vector).
//

template < class F, int tagI=0 >
class MinArgMinReduction {

    static const int tagJ = 1-tagI;

  public :

    struct sEval { // static wrapper
        using VARSI = typename F::template VARS<tagI>; // Use the tag to select the "parallel"  variable
        using VARSJ = typename F::template VARS<tagJ>; // Use the tag to select the "summation" variable
        using VARSP = typename F::template VARS<2>;

        using DIMSX = typename GetDims<VARSI>::template PUTLEFT<F::DIM>; // dimensions of "i" variables. We add the output's dimension.
        using DIMSY = GetDims<VARSJ>;                           // dimensions of "j" variables
        using DIMSP = GetDims<VARSP>;                           // dimensions of parameters variables
        
        static const int DIM = 2*F::DIM;		// DIM is dimension of output of convolution ; for a argmin reduction it is equal to the dimension of output of formula

	static const int DIMRED = DIM;	// dimension of temporary variable for reduction
		
        using FORM  = F;  // We need a way to access the actual function being used. 
        // using FORM  = AutoFactorize<F>;  // alternative : auto-factorize the formula (see factorize.h file)
        // remark : using auto-factorize should be the best to do but it may slow down the compiler a lot..
        
        using INDSI = GetInds<VARSI>;
        using INDSJ = GetInds<VARSJ>;
        using INDSP = GetInds<VARSP>;

        using INDS = ConcatPacks<ConcatPacks<INDSI,INDSJ>,INDSP>;  // indices of variables
        static_assert(CheckAllDistinct<INDS>::val,"Incorrect formula : at least two distinct variables have the same position index.");
        
        static const int NMINARGS = 1+INDS::MAX; // minimal number of arguments when calling the formula. 

		template < typename TYPE >
		struct InitializeReduction {
			HOST_DEVICE INLINE void operator()(TYPE *tmp) {
				for(int k=0; k<F::DIM; k++)
					tmp[k] = PLUS_INFINITY<TYPE>::value; // initialize output
				for(int k=F::DIM; k<DIM; k++)
					tmp[k] = 0; // initialize output
			}
		};

        template < typename... Args >
        HOST_DEVICE INLINE void operator()(Args... args) {
            F::template Eval<INDS>(args...);
        }
        
		// equivalent of the += operation
		template < typename TYPE >
		struct ReducePairShort {
			HOST_DEVICE INLINE void operator()(TYPE *tmp, TYPE *xi, int j) {
				for(int k=0; k<F::DIM; k++) {
					if(xi[k]<tmp[k]) {
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
					if(xi[k]<tmp[k]) {
						tmp[k] = xi[k];
						tmp[F::DIM+k] = xi[F::DIM+k];
					}
				}
			}
		};
        
		template < typename TYPE >
		struct FinalizeOutput {
			HOST_DEVICE INLINE void operator()(TYPE *tmp, TYPE *out) {
				for(int k=0; k<DIM; k++)
					out[k] = tmp[k];
			}
		};
		
		template < class V, class GRADIN >
		using DiffT = Zero<V::DIM>;
        
    };

};

}
