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
class ZeroReduction : public Reduction<Zero<DIM>,tagI> {

  public :
          		
        template < class CONV, typename TYPE, typename... Args >
        static void Eval(int nx, int ny, TYPE *out, Args... args) {
        	for(int k=0; k<(tagI==0?nx:ny)*DIM; k++)
        		 out[k] = 0;
        }
                
		template < class V, class GRADIN >
		using DiffT = ZeroReduction<V::DIM,V::CAT>;
                
};

}
