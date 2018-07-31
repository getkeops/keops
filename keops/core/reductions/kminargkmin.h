#pragma once

#include "core/Pack.h"

#include "core/autodiff.h"

#include "core/reductions/reduction.h"

namespace keops {
// Implements the k-min-arg-k-min reduction operation : for each i or each j, find the values and indices of the
// k minimal values of Fij
// operation is vectorized: if Fij is vector-valued, arg-k-min is computed for each dimension.
// tagI is equal:
// - to 0 if you do the reduction over j (with i the index of the output vector),
// - to 1 if you do the reduction over i (with j the index of the output vector).
//

template < class F, int K, int tagI=0 >
class KMinArgKMinReduction : public Reduction<F,tagI> {

  public :
        
        static const int DIM = 2*K*F::DIM;		// DIM is dimension of output of convolution ; for a arg-k-min reduction it is equal to the dimension of output of formula

	static const int DIMRED = DIM;	// dimension of temporary variable for reduction
		
        template < class CONV, typename... Args >
        static void Eval(Args... args) {
        	CONV::Eval(KMinArgKMinReduction<F,K,tagI>(),args...);
        }
                
		template < typename TYPE >
		struct InitializeReduction {
			HOST_DEVICE INLINE void operator()(TYPE *tmp) {
				for(int k=0; k<F::DIM; k++) {
					for(int l=k; l<K*2*F::DIM+k; l+=2*F::DIM) {
						tmp[l] = PLUS_INFINITY<TYPE>::value; // initialize output
						tmp[l+F::DIM] = 0; // initialize output
					}
				}
			}
		};
		

		// equivalent of the += operation
		template < typename TYPE >
		struct ReducePairShort {
			HOST_DEVICE INLINE void operator()(TYPE *tmp, TYPE *xi, int j) {
				TYPE xik;
				int l;
				for(int k=0; k<F::DIM; k++) {					
					xik = xi[k];
					for(l=(K-1)*2*F::DIM+k; l>=k && xik<tmp[l]; l-=2*F::DIM) { 
						TYPE tmpl = tmp[l];
						int indtmpl = tmp[l+F::DIM];
						tmp[l] = xik;
						tmp[l+F::DIM] = j;
						if(l<(K-1)*2*F::DIM+k) {
							tmp[l+2*F::DIM] = tmpl;
							tmp[l+2*F::DIM+F::DIM] = indtmpl;
						}
					}
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
		
		// no gradient implemented here		        

};

}
