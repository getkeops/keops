#pragma once

#include <sstream>

#include "core/Pack.h"

#include "core/autodiff.h"

#include "core/reductions/reduction.h"

// Implements the LogSumExp reduction operation
// tagI is equal:
// - to 0 if you do the summation over j (with i the index of the output vector),
// - to 1 if you do the summation over i (with j the index of the output vector).
// Giving a "LogSumExp" to a Conv1D/2D routine will automatically
// result in it using a numerically stable reduce operation.

namespace keops {

template < int DIM >
using LSEFIN = Add<_X<0,1>,Log<_X<1,DIM>>>;

template < class F, class G_=IntConstant<1>, class FIN_=LSEFIN<G_::DIM>, class GRADIN_=Dummy, int tagI=0 >
class LogSumExpReduction : public Reduction<Concat<Concat<F,G_>,GRADIN_>,tagI> {

  public :
  
  		using G = G_;
  		using FIN = FIN_;
  		
  		static const int INDGRADIN = GRADIN_::N;

		using PARENT = Reduction<Concat<Concat<F,G_>,GRADIN_>,tagI>;

		static const int DIM = FIN::DIM;
		
		static_assert(F::DIM==1,"LogSumExp requires first formula F of dimension 1.");
		
		static const int DIMRED = G::DIM + F::DIM;				// dimension of temporary variable for reduction
		
        template < class CONV, typename... Args >
        static int Eval(Args... args) {
        	return CONV::Eval(LogSumExpReduction<F,G,FIN,GRADIN_,tagI>(),args...);
        }
                
		template < typename TYPE >
		struct InitializeReduction {
			HOST_DEVICE INLINE void operator()(TYPE *tmp) {
				// We fill empty cells with the neutral element of the reduction operation,
				//                   (-inf,0) = e^{-inf} * 0 = 0
				
				// We should use 0xfff0000000000000 for doubles
				//-340282346638528859811704183484516925440.0f;//__int_as_float(0xff800000); // -infty, as +infty = 0x7f800000
				tmp[0] = NEG_INFINITY<TYPE>::value;
				for(int k=1; k<DIMRED; k++)
					tmp[k] = 0.0f;
			}
		};

		// equivalent of the += operation
		template < typename TYPE >
		struct ReducePairShort {
			HOST_DEVICE INLINE void operator()(TYPE *tmp, TYPE *xi, int j) {
				// (m,s) + (m',s'), i.e. exp(m)*s + exp(m')
				TYPE tmpexp;
				if(tmp[0] > xi[0]) { // =  exp(m)  * (s + s'*exp(m'-m))   if m > m'
					tmpexp = exp( xi[0]-tmp[0] );
					for(int k=1; k<DIMRED; k++)
						tmp[k] += xi[k]*tmpexp ;
				} else {             // =  exp(m') * (s' + exp(m-m')*s)   if m <= m'
					tmpexp = exp( tmp[0]-xi[0] );
					for(int k=1; k<DIMRED; k++)
						tmp[k] = xi[k] + tmpexp * tmp[k] ;
					tmp[0] = xi[0] ;
				}
			}
		};
                
		// equivalent of the += operation
		template < typename TYPE >
		struct ReducePair {
			HOST_DEVICE INLINE void operator()(TYPE *tmp, TYPE *xi) {
				// (m,s) + (m',s'), i.e. exp(m)*s + exp(m')
				TYPE tmpexp;
				if(tmp[0] > xi[0]) { // =  exp(m)  * (s + s'*exp(m'-m))   if m > m'
					tmpexp = exp( xi[0]-tmp[0] );
					for(int k=1; k<DIMRED; k++)
						tmp[k] += xi[k]*tmpexp ;
				} else {             // =  exp(m') * (s' + exp(m-m')*s)   if m <= m'
					tmpexp = exp( tmp[0]-xi[0] );
					for(int k=1; k<DIMRED; k++)
						tmp[k] = xi[k] + tmpexp * tmp[k] ;
					tmp[0] = xi[0] ;
				}
			}
		};

		static const int LOCINDGRADIN = 1+INTMIN<IndVal<typename PARENT::INDSI,INDGRADIN>::value,PARENT::INDSI::SIZE-1>::value;
                
		template < typename TYPE >
		struct FinalizeOutput {
			HOST_DEVICE INLINE void operator()(TYPE *tmp, TYPE *out, TYPE **px, int i) {
			TYPE *tmp2 = px[LOCINDGRADIN];
            		FIN::template Eval<pack<0,1,2>>(out,tmp,tmp+1,tmp2+i*GRADIN_::DIM);
			}
		};
        
        	template < class GRADIN >
		using A = Grad<FIN,_X<1,G::DIM>,GRADIN>;
		
		template < class V, class GRADIN >
		using B = typename A<GRADIN>::template Replace<_X<1,G::DIM>,Extract<_X<1,(1+V::DIM)*G::DIM>,0,G::DIM>>;
		
		template < class V, class GRADIN >
		using C = MatVecMult<Extract<_X<1,(1+V::DIM)*G::DIM>,G::DIM,V::DIM*G::DIM>,B<V,GRADIN>>;
				
		template < class V >
		using D = Add< TensorProd<GradMatrix<F,V>,G> , GradMatrix<G,V> > ;		
		
		template < class V, class GRADIN >
		using DiffT = LogSumExpReduction<F,Concat<G,D<V>>,C<V,_X<2,GRADIN::DIM>>,GRADIN,V::CAT>;
		
};


}
