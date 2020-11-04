
namespace keops {

//////////////////////////////////////////////////////////////
////         DIFFCLAMPINT : DiffClampInt< F, A, B >       ////
//////////////////////////////////////////////////////////////

// DiffClampInt(x,a,b) = 0 if x<a, 1 if a<=x<=b, 0 if b<x 
// N.B. used as derivative of ClampInt operation

template<class F, int A, int B>
struct DiffClampInt : VectorizedScalarUnaryOp<DiffClampInt, F, A, B> {

  static void PrintIdString(::std::stringstream &str) { str << "DiffClampInt"; }

  template < typename TYPE > 
  struct Operation_Scalar {
	DEVICE INLINE void operator() (TYPE &out, TYPE &outF) {
    	  out = keops_diffclampint(outF, A, B);
    }
  };

  // N.B.   ClampInt(F,A,B) = ReLU(F-A) + A - ReLU(F-B)
  // We use this fact to avoid writing another custom operation for the gradient.
  // (This may be slower however...)

  //using Generic_ClampInt = Subtract<Add<IntConstant<A>,ReLU<Subtract<F,IntConstant<A>>>>,ReLU<Subtract<F,IntConstant<B>>>>;

  //template<class V, class GRADIN>
  //using DiffT = typename Generic_ClampInt::template DiffT<V,GRADIN>;
  
  template<class V, class GRADIN>
  using DiffT = Zero<V::DIM>;

};

}
