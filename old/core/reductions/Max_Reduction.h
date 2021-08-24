#pragma once

#include <sstream>

#include "core/autodiff/UnaryOp.h"
#include "core/reductions/Reduction.h"
#include "core/reductions/Zero_Reduction.h"
#include "core/pre_headers.h"
#include "core/utils/Infinity.h"
#include "core/utils/TypesUtils.h"

namespace keops {
	
// Implements the max reduction operation : for each i or each j, find the
// maximal value of Fij
// operation is vectorized: if Fij is vector-valued, max is computed for each dimension.

template < class F, int tagI=0 >
struct Max_Reduction : public Reduction<F,tagI>, UnaryOp<Max_Reduction,F,tagI> {

    static const int DIM = F::DIM;		// DIM is dimension of output of convolution ; for a max reduction it is equal to the dimension of output of formula

	static const int DIMRED = F::DIM;	// dimension of temporary variable for reduction
		
    static void PrintIdString(::std::stringstream& str) {
        str << "Max_Reduction";
    }

		template < typename TYPEACC, typename TYPE >
		struct InitializeReduction {
			DEVICE INLINE void operator()(TYPEACC *tmp) {
				VectAssign<F::DIM>(tmp, NEG_INFINITY<TYPE>::value);
			}
		};

		template < typename TYPEACC, typename TYPE >
		struct ReducePairScalar {
			DEVICE INLINE void operator()(TYPEACC &tmp, const TYPE &xi) {
					if(xi>tmp) {
						tmp = xi;
					}
				}
			};

#if USE_HALF && GPU_ON
template < typename TYPEACC >
	struct ReducePairScalar<TYPEACC, half2 > {
			DEVICE INLINE void operator()(TYPEACC &tmp, const half2 &xi) {
					half2 cond = __hgt2(xi,tmp);
					half2 negcond = cast_to<half2>(1.0f)-cond;
					tmp = cond * xi + negcond * tmp;
				}
			};
#endif

		// equivalent of the += operation
		template < typename TYPEACC, typename TYPE >
		struct ReducePairShort {
			DEVICE INLINE void operator()(TYPEACC *tmp, TYPE *xi, TYPE val) {
				VectApply<ReducePairScalar<TYPEACC,TYPE>,F::DIM,F::DIM>(tmp,xi);
			}
		};
        
		// equivalent of the += operation
		template < typename TYPEACC, typename TYPE >
		struct ReducePair {
			DEVICE INLINE void operator()(TYPEACC *tmp, TYPE *xi) {
				VectApply<ReducePairScalar<TYPEACC,TYPE>,F::DIM,F::DIM>(tmp,xi);
			}
		};
        
	    template < typename TYPEACC, typename TYPE >
	    struct FinalizeOutput {
	        DEVICE INLINE void operator()(TYPEACC *acc, TYPE *out, int i) {
				VectCopy<F::DIM>(out,acc);
	        }
	    };

	    // no gradient implemented here
		
};


/////////////////////////////////////////////////////////////////////////
//          max+argmax reduction : base class                          //
/////////////////////////////////////////////////////////////////////////
template < class F, int tagI=0 >
struct Max_ArgMax_Reduction_Base : public Reduction<F,tagI> {

    // We work with a (values,indices) vector
	static const int DIMRED = 2*F::DIM;	// dimension of temporary variable for reduction
		
		template < typename TYPEACC, typename TYPE >
		struct InitializeReduction {
			DEVICE INLINE void operator()(TYPEACC *tmp) {
				VectAssign<F::DIM>(tmp, NEG_INFINITY<TYPE>::value);
				VectAssign<F::DIM>(tmp + F::DIM, 0.0f);
			}
		};

template < typename TYPEACC, typename TYPE >
		struct ReducePairScalar {
			DEVICE INLINE void operator()(TYPEACC &tmpval, TYPEACC &tmpind, TYPE &xi, TYPE &ind) {
					if(xi>tmpval) {
						tmpval = xi;
						tmpind = ind;
					}
				}
			};

#if USE_HALF && GPU_ON
template < typename TYPEACC >
	struct ReducePairScalar<TYPEACC, half2 > {
			DEVICE INLINE void operator()(TYPEACC &tmpval, TYPEACC &tmpind, half2 &xi, half2 &ind) {
					half2 cond = __hgt2(xi,tmpval);
					half2 negcond = cast_to<half2>(1.0f)-cond;
					tmpval = cast_to<TYPEACC> (cond * xi + negcond * tmpval);
					tmpind = cast_to<TYPEACC> (cond * ind + negcond * tmpind);
				}
			};
#endif

		// equivalent of the += operation
		template < typename TYPEACC, typename TYPE >
		struct ReducePairShort {
			DEVICE INLINE void operator()(TYPEACC *tmp, TYPE *xi, TYPE ind) {
				VectApply<ReducePairScalar<TYPEACC,TYPE>,F::DIM,F::DIM,F::DIM,1>(tmp,tmp+F::DIM,xi,&ind);
			}
		};
        
		// equivalent of the += operation
		template < typename TYPEACC, typename TYPE >
		struct ReducePair {
			DEVICE INLINE void operator()(TYPEACC *tmp, TYPE *xi) {
				VectApply<ReducePairScalar<TYPEACC,TYPE>,F::DIM,F::DIM,F::DIM,F::DIM>(tmp,tmp+F::DIM,xi,xi+F::DIM);
			}
		};
        
};


// Implements the max+argmax reduction operation : for each i or each j, find the maximal value of Fij and its index
// operation is vectorized: if Fij is vector-valued, max+argmax is computed for each dimension.

template < class F, int tagI=0 >
struct Max_ArgMax_Reduction : public Max_ArgMax_Reduction_Base<F,tagI>, UnaryOp<Max_ArgMax_Reduction,F,tagI> {

        static const int DIM = 2*F::DIM;		// DIM is dimension of output of convolution ; for a max-argmax reduction it is equal to 2 times the dimension of output of formula
		
    static void PrintIdString(::std::stringstream& str) {
        str << "Max_ArgMax_Reduction";
    }
        
    template < typename TYPEACC, typename TYPE >
    struct FinalizeOutput {
        DEVICE INLINE void operator()(TYPEACC *acc, TYPE *out, int i) {
			VectCopy<DIM>(out,acc);
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
        DEVICE INLINE void operator()(TYPEACC *acc, TYPE *out, int i) {
			VectCopy<F::DIM>(out,acc+F::DIM);
        }
    };

    template < class V, class GRADIN >
    using DiffT = Zero_Reduction<V::DIM,(V::CAT)%2>;
    // remark : if V::CAT is 2 (parameter), we will get tagI=(V::CAT)%2=0, so we will do reduction wrt j.
    // In this case there is a summation left to be done by the user.

};

#define ArgMax_Reduction(F,I) KeopsNS<ArgMax_Reduction<decltype(InvKeopsNS(F)),I>>()
#define Max_Reduction(F,I) KeopsNS<Max_Reduction<decltype(InvKeopsNS(F)),I>>()
#define Max_ArgMax_Reduction(F,I) KeopsNS<Max_ArgMax_Reduction<decltype(InvKeopsNS(F)),I>>()

}
