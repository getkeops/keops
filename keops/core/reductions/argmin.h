#pragma once

#include "core/Pack.h"

#include "core/autodiff.h"

#include "core/reductions/reduction.h"
#include "core/reductions/zero.h"

namespace keops {
// Implements the argmin reduction operation : for each i or each j, find the index of the
// minimal value of Fij
// operation is vectorized: if Fij is vector-valued, argmin is computed for each dimension.
// tagI is equal:
// - to 0 if you do the reduction over j (with i the index of the output vector),
// - to 1 if you do the reduction over i (with j the index of the output vector).
//

template < class F, int tagI=0 >
class ArgMinReduction : public Reduction<F,tagI> {

  public :
        
        static const int DIM = F::DIM;		// DIM is dimension of output of convolution ; for a argmin reduction it is equal to the dimension of output of formula

	static const int DIMRED = 2*DIM;	// dimension of temporary variable for reduction
		
        template < class CONV, typename... Args >
        static int Eval(Args... args) {
        	return CONV::Eval(ArgMinReduction<F,tagI>(),args...);
        }
                
		template < typename TYPE >
		struct InitializeReduction {
			HOST_DEVICE INLINE void operator()(TYPE *tmp) {
				for(int k=0; k<DIM; k++)
					tmp[k] = PLUS_INFINITY<TYPE>::value; // initialize output
				for(int k=DIM; k<2*DIM; k++)
					tmp[k] = 0; // initialize output
			}
		};

		// equivalent of the += operation
		template < typename TYPE >
		struct ReducePairShort {
			HOST_DEVICE INLINE void operator()(TYPE *tmp, TYPE *xi, int j) {
				for(int k=0; k<DIM; k++) {
					if(xi[k]<tmp[k]) {
						tmp[k] = xi[k];
						tmp[DIM+k] = j;
					}
				}
			}
		};
        
		// equivalent of the += operation
		template < typename TYPE >
		struct ReducePair {
			HOST_DEVICE INLINE void operator()(TYPE *tmp, TYPE *xi) {
				for(int k=0; k<DIM; k++) {
					if(xi[k]<tmp[k]) {
						tmp[k] = xi[k];
						tmp[DIM+k] = xi[DIM+k];
					}
				}
			}
		};
        
		template < typename TYPE >
		struct FinalizeOutput {
			HOST_DEVICE INLINE void operator()(TYPE *tmp, TYPE *out, TYPE **px, int i) {
				for(int k=0; k<DIM; k++)
            		out[k] = tmp[DIM+k];
			}
		};
		
		template < class V, class GRADIN >
		using DiffT = ZeroReduction<V::DIM,V::CAT>;
        

};

}
