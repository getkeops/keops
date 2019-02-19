#pragma once

#include "core/Pack.h"

#include "core/autodiff.h"
#include "core/formulas/maths.h"

namespace keops {

// Default class for the reduction operation. Only derived classes can do something.
// tagI is equal:
// - to 0 if you do the reduction over j (with i the index of the output vector),
// - to 1 if you do the reduction over i (with j the index of the output vector).

template < class F_, int tagI_=0 >
struct Reduction {

    using F = F_;

    static const int tagI = tagI_;
    static_assert(tagI==0 || tagI==1, "tagI should be 0 or 1 in for reduction operation.");
    static const int tagJ = 1-tagI;

    static const int CAT = tagI; // category of the output vector (used in alias GradFromPos, see autodiff.h)

    using VARSI = typename F::template VARS<tagI>; // Use the tag to select the "parallel"  variable
    using VARSJ = typename F::template VARS<tagJ>; // Use the tag to select the "summation" variable
    using VARSP = typename F::template VARS<2>;    // Parameters

    using DIMSX = typename GetDims<VARSI>::template PUTLEFT<F::DIM>; // dimensions of "i" variables. We add the output's dimension.
    using DIMSY = GetDims<VARSJ>;                           // dimensions of "j" variables
    using DIMSP = GetDims<VARSP>;                           // dimensions of parameters variables

    using FORM  = F;  // We need a way to access the actual function being used.
    // using FORM  = AutoFactorize<F>;  // alternative : auto-factorize the formula (see factorize.h file)
    // remark : using auto-factorize should be the best to do but it may slow down the compiler a lot..

    using INDSI = GetInds<VARSI>;
    using INDSJ = GetInds<VARSJ>;
    using INDSP = GetInds<VARSP>;

    using INDS = ConcatPacks<ConcatPacks<INDSI,INDSJ>,INDSP>;  // indices of variables
    static_assert(CheckAllDistinct<INDS>::val,"Incorrect formula : at least two distinct variables have the same position index.");

    static const int NMINARGS = 1+INDS::MAX; // minimal number of arguments when calling the formula.

    template < typename... Args >
    HOST_DEVICE INLINE void operator()(Args... args) {
        F::template Eval<INDS>(args...);
    }
    
};

// default evaluation by calling Cpu/Gpu reduction engine, taking care of axis of reduction
template < class RED, class MODE >
struct Eval {
	template < typename... Args >
	static int Run(int nx, int ny, Args... args) {
		if(RED::tagI==0)
       			return MODE::Eval(RED(),nx,ny,args...);
		else if(RED::tagI==1)
       			return MODE::Eval(RED(),ny,nx,args...);
       	else return -1;
	}
};

}
