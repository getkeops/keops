# pragma once

#include "core/pack/Pack.h"

namespace keops {


// call : calls a function, unrolling dimensions to get correct inputs
// Example:
//   call< pack<5,1,3> >(fun,out,x);
// will execute
//   fun(out,x,x+5,x+6)
// N.B. the last dimension, 3 in the example, is not used. But the 
// assumption is that x+6 corresponds to a vector of size 3
  
template < class DIMS >
struct call_Impl {
  template < typename TYPE, class FUN, typename... Args >
  HOST_DEVICE static void Eval(FUN fun, TYPE *out, TYPE *x, Args... args) {
    fun(out, args...);
  }
};

template < int FIRST, int... NEXTS >
struct call_Impl < pack<FIRST,NEXTS...> > {
  using NEXT = pack<NEXTS...>;
  template < typename TYPE, class FUN, typename... Args >
  HOST_DEVICE static void Eval(FUN fun, TYPE *out, TYPE *x, Args... args) {
    call_Impl<NEXT>::Eval(fun, out, x + FIRST, args..., x);         // Append [x[0:FIRST]] to the list of arguments, then iterate.
  }
};

template < class DIMS, typename TYPE, class FUN, typename... Args >
HOST_DEVICE static void call(FUN fun, TYPE *out, Args... args) {
  call_Impl<DIMS>::Eval(fun, out, args...); 
}



// Version with two packs of dimensions
// Example:
//   call< pack<5,1,3>, pack<2,3,3,1> >(fun,out,x,y);
// will execute
//   fun(out,x,x+5,x+6,y,y+2,y+5,y+8)
  
template < class DIMS1, class DIMS2 >
struct call2_Impl {
  template < typename TYPE, class FUN, typename... Args >
  HOST_DEVICE static void Eval(FUN fun, TYPE *out, TYPE *x, Args... args) {
    call_Impl<DIMS2>::Eval(fun, out, args...);
  }
};

template < class DIMS2, int FIRST, int... NEXTS >
struct call2_Impl < pack<FIRST,NEXTS...>, DIMS2 > {
  using NEXT = pack<NEXTS...>;
  template < typename TYPE, class FUN, typename... Args >
  HOST_DEVICE static void Eval(FUN fun, TYPE *out, TYPE *x, Args... args) {
    call2_Impl<NEXT,DIMS2>::Eval(fun, out, x + FIRST, args..., x);
  }
};

template < class DIMS1, class DIMS2, typename TYPE, class FUN, typename... Args >
HOST_DEVICE static void call(FUN fun, TYPE *out, Args... args) {
  call2_Impl<DIMS1,DIMS2>::Eval(fun, out, args...); 
}



// version with three packs of dimensions
// Example:
//   call< pack<5,1,3>, pack<2,3>, pack<1,2,3,4> >(fun,out,x,y,z);
// will execute
//   fun(out,x,x+5,x+6,y,y+2,z,z+1,z+3,z+6)
  
template < class DIMS1, class DIMS2, class DIMS3 >
struct call3_Impl {
  template < typename TYPE, class FUN, typename... Args >
  HOST_DEVICE static void Eval(FUN fun, TYPE *out, TYPE *x, Args... args) {
    call2_Impl<DIMS2,DIMS3>::Eval(fun, out, args...);
  }
};

template < class DIMS2, class DIMS3, int FIRST, int... NEXTS >
struct call3_Impl < pack<FIRST,NEXTS...>, DIMS2, DIMS3 > {
  using NEXT = pack<NEXTS...>;
  template < typename TYPE, class FUN, typename... Args >
  HOST_DEVICE static void Eval(FUN fun, TYPE *out, TYPE *x, Args... args) {
    call3_Impl<NEXT,DIMS2,DIMS3>::Eval(fun, out, x + FIRST, args..., x);
  }
};

template < class DIMS1, class DIMS2, class DIMS3, typename TYPE, class FUN, typename... Args >
HOST_DEVICE static void call(FUN fun, TYPE *out, Args... args) {
  call3_Impl<DIMS1,DIMS2,DIMS3>::Eval(fun, out, args...); 
}



// Now with four packs of dimensions
  
template < class DIMS1, class DIMS2, class DIMS3, class DIMS4 >
struct call4_Impl {
  template < typename TYPE, class FUN, typename... Args >
  HOST_DEVICE static void Eval(FUN fun, TYPE *out, TYPE *x, Args... args) {
    call3_Impl<DIMS2,DIMS3,DIMS4>::Eval(fun, out, args...);
  }
};

template < class DIMS2, class DIMS3, class DIMS4, int FIRST, int... NEXTS >
struct call4_Impl < pack<FIRST,NEXTS...>, DIMS2, DIMS3, DIMS4 > {
  using NEXT = pack<NEXTS...>;
  template < typename TYPE, class FUN, typename... Args >
  HOST_DEVICE static void Eval(FUN fun, TYPE *out, TYPE *x, Args... args) {
    call4_Impl<NEXT,DIMS2,DIMS3,DIMS4>::Eval(fun, out, x + FIRST, args..., x);
  }
};

template < class DIMS1, class DIMS2, class DIMS3, class DIMS4, typename TYPE, class FUN, typename... Args >
HOST_DEVICE static void call(FUN fun, TYPE *out, Args... args) {
  call4_Impl<DIMS1,DIMS2,DIMS3,DIMS4>::Eval(fun, out, args...); 
}


}
