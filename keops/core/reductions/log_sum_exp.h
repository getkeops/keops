#pragma once

#include <sstream>

#include "core/Pack.h"

#include "core/autodiff.h"

#ifdef __CUDACC__
	#include <npp.h>
#endif

// Implements the LogSumExp reduction operation
// tagI is equal:
// - to 0 if you do the summation over j (with i the index of the output vector),
// - to 1 if you do the summation over i (with j the index of the output vector).
// Giving a "LogSumExp" to a Conv1D/2D routine will automatically
// result in it using a numerically stable reduce operation.

namespace keops {

template <typename TYPE>
struct NEG_INFINITY;

template <>
struct NEG_INFINITY<float> {
    static constexpr float value = - INFINITY_FLOAT;
};

template <>
struct NEG_INFINITY<double> {
    static constexpr double value = - INFINITY_DOUBLE;
};

template < class F, int tagI=0 >
class LogSumExpReduction {

    static const int tagJ = 1-tagI;

  public :

    struct sEval { // static wrapper
        using VARSI = typename F::template VARS<tagI>; // Use the tag to select the "parallel"  variable
        using VARSJ = typename F::template VARS<tagJ>; // Use the tag to select the "summation" variable
        using VARSP = typename F::template VARS<2>;

        using DIMSX = typename GetDims<VARSI>::template PUTLEFT<F::DIM>; // dimensions of "i" variables. We add the output's dimension.
        using DIMSY = GetDims<VARSJ>;                           // dimensions of "j" variables
        using DIMSP = GetDims<VARSP>;                           // dimensions of parameters variables

		static const int DIM = F::DIM;
		
		static_assert(1==DIM,"LogSumExp is only implemented for scalars.");
		
		static const int DIMRED = 2 * DIM;									// dimension of temporary variable for reduction
		
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
				// We fill empty cells with the neutral element of the reduction operation,
				//                   (-inf,0) = e^{-inf} * 0 = 0
				
				// We should use 0xfff0000000000000 for doubles
				//-340282346638528859811704183484516925440.0f;//__int_as_float(0xff800000); // -infty, as +infty = 0x7f800000
				tmp[0] = NEG_INFINITY<TYPE>::value;
				tmp[1] = 0.0f;
			}
		};

        template < typename... Args >
        HOST_DEVICE INLINE void operator()(Args... args) {
            F::template Eval<INDS>(args...);
        }
        
		// equivalent of the += operation
		template < typename TYPE >
		struct ReducePair {
			HOST_DEVICE INLINE void operator()(TYPE *tmp, TYPE *xi, int j) {
				// (m,s) + (m',1), i.e. exp(m)*s + exp(m')
				if(tmp[0] > xi[0]) { // =  exp(m)  * (s + exp(m'-m))   if m > m'
					tmp[1] += exp( xi[0]-tmp[0] ) ;
				} else {             // =  exp(m') * (1 + exp(m-m')*s)   if m <= m'
					tmp[1] = 1.0 + exp( tmp[0]-xi[0] ) * tmp[1] ;
					tmp[0] = xi[0] ;
				}
			}
		};
        
		template < typename TYPE >
		struct FinalizeOutput {
			HOST_DEVICE INLINE void operator()(TYPE *tmp, TYPE *out) {
            		out[0] = tmp[0] + log(tmp[1]);
			}
		};
        
		template < class V, class GRADIN >
		using DiffT = typename SumReduction<Grad<F,V,GRADIN>,V::CAT>::sEval;
        
    };

};


}
