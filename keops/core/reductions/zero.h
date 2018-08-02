#pragma once

#include "core/Pack.h"

#include "core/autodiff.h"

#include "core/reductions/reduction.h"

namespace keops {

// Implements the zero reduction operation (fills output with zeros)
// tagI is equal:
// - to 0 if you do the summation over j (with i the index of the output vector),
// - to 1 if you do the summation over i (with j the index of the output vector).
//
template < int DIM, int tagI=0 >
struct ZeroReduction : public Reduction<Zero<DIM>,tagI> {
	
	template < class V, class GRADIN >
	using DiffT = ZeroReduction<V::DIM,(V::CAT)%2>;
 	// remark : if V::CAT is 2 (parameter), we will get tagI=(V::CAT)%2=0, so we will do reduction wrt j. 
	// In this case there is a summation left to be done by the user.
                
};

// specialized evaluation : no need to call a reduction operation for filling zeros

template < int DIM, int tagI, class MODE >
struct Eval<ZeroReduction<DIM,tagI>,MODE> {
	template < typename TYPE, typename... Args >
	static int Run(int nx, int ny, TYPE *out, Args... args) {
        	for(int k=0; k<(tagI==0?nx:ny)*DIM; k++)
        		 out[k] = 0;
		return 0;
	}
};

}
