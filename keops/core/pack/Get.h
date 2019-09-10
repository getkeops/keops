#pragma once

namespace keops {



// custom implementation of tuple "get" function, to avoid the use of thrust/tuple
// which is limited to 10 arguments. This function works only when all arguments
// have same type

template < int N, typename TYPE >
struct GetHelper {
  template < typename... ARGS >
  static HOST_DEVICE INLINE TYPE Eval(TYPE arg0, ARGS... args) {
    return GetHelper<N-1,TYPE>::Eval(args...);
  }
};

template < typename TYPE >
struct GetHelper<0,TYPE> {
  template < typename... ARGS >
  static HOST_DEVICE INLINE TYPE Eval(TYPE arg0, ARGS... args) {
    return arg0;
  }
};

template < int N, typename TYPE, typename... ARGS >
static HOST_DEVICE INLINE TYPE Get(TYPE arg0, ARGS... args) {
  return GetHelper<N,TYPE>::Eval(arg0, args...);
}
}