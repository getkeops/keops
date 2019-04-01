#pragma once

#include "core/Pack.h"

#include "core/autodiff.h"

#include "core/reductions/zero.h"

namespace keops {
// Implements the k-min-arg-k-min reduction operation : for each i or each j, find the values and indices of the
// k minimal values of Fij
// operation is vectorized: if Fij is vector-valued, arg-k-min is computed for each dimension.

template < class F, int K, int tagI=0 >
struct KMin_ArgKMin_Reduction : public Reduction<F,tagI> {

    static const int DIM = 2*K*F::DIM;		// DIM is dimension of output of convolution ; for a arg-k-min reduction it is equal to the dimension of output of formula

	static const int DIMRED = DIM;	// dimension of temporary variable for reduction
		
    static void PrintId(std::stringstream& str) {
        str << "KMin_ArgKMin_Reduction(";			// prints "("
        F::PrintId(str);				// prints the formula F
        str << ",K=" << K << ",tagI=" << tagI << ")";
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

	// equivalent of the += operation
	template < typename TYPE >
	struct ReducePair {
		HOST_DEVICE INLINE void operator()(TYPE *tmp, TYPE *xi) {
		    TYPE out[DIMRED];
			for(int k=0; k<F::DIM; k++) {
			    int p = k;
			    int q = k;
			    for(int l=k; l<DIMRED; l+=2*F::DIM) {
			        if(xi[p]<tmp[q]) {
					    out[l] = xi[p];
					    out[F::DIM+l] = xi[F::DIM+p];
					    p += 2*F::DIM;
					}
					else {
					    out[l] = tmp[q];
					    out[F::DIM+l] = tmp[F::DIM+q];
					    q += 2*F::DIM;
					}  
				}
			}
			for(int k=0; k<DIMRED; k++)
			    tmp[k] = out[k];
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

// Implements the arg-k-min reduction operation : for each i or each j, find the indices of the
// k minimal values of Fij
// operation is vectorized: if Fij is vector-valued, arg-k-min is computed for each dimension.

template < class F, int K, int tagI=0 >
struct ArgKMin_Reduction : public KMin_ArgKMin_Reduction<F,K,tagI> {


    static const int DIM = K*F::DIM;		// DIM is dimension of output of convolution ; for a arg-k-min reduction it is equal to the dimension of output of formula

    static void PrintId(std::stringstream& str) {
        str << "ArgKMin_Reduction(";			// prints "("
        F::PrintId(str);				// prints the formula F
        str << ",K=" << K << ",tagI=" << tagI << ")";
    }
                  
    template < typename TYPE >
    struct FinalizeOutput {
        HOST_DEVICE INLINE void operator()(TYPE *tmp, TYPE *out, TYPE **px, int i) {
            for(int k=0; k<F::DIM; k++)
                for(int p=k, l=k; l<K*2*F::DIM+k; p+=F::DIM, l+=2*F::DIM)
                    out[p] = tmp[l+F::DIM];
        }
    };

    template < class V, class GRADIN >
    using DiffT = Zero_Reduction<V::DIM,(V::CAT)%2>;
    // remark : if V::CAT is 2 (parameter), we will get tagI=(V::CAT)%2=0, so we will do reduction wrt j.
    // In this case there is a summation left to be done by the user.


};

// Implements the k-min reduction operation : for each i or each j, find the
// k minimal values of Fij
// operation is vectorized: if Fij is vector-valued, arg-k-min is computed for each dimension.

template < class F, int K, int tagI=0 >
struct KMin_Reduction : public KMin_ArgKMin_Reduction<F,K,tagI> {


        static const int DIM = K*F::DIM;		// DIM is dimension of output of convolution ; for a arg-k-min reduction it is equal to the dimension of output of formula
                 
    static void PrintId(std::stringstream& str) {
        str << "KMin_Reduction(";			// prints "("
        F::PrintId(str);				// prints the formula F
        str << ",K=" << K << ",tagI=" << tagI << ")";
    }

    template < typename TYPE >
    struct FinalizeOutput {
        HOST_DEVICE INLINE void operator()(TYPE *tmp, TYPE *out, TYPE **px, int i) {
            for(int k=0; k<F::DIM; k++)
                for(int p=k, l=k; l<K*2*F::DIM+k; p+=F::DIM, l+=2*F::DIM)
                    out[p] = tmp[l];
        }
    };

    // no gradient implemented here


};

}
