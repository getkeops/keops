#pragma once

#include <sstream>

#include "core/autodiff/VectorizedScalarTernaryOp.h"
#include "core/utils/keops_math.h"

namespace keops {

template <class F, class G, class H>
struct IfElse : VectorizedScalarTernaryOp<IfElse, F, G, H> {

  static void PrintIdString(::std::stringstream &str) { str << "IfElse"; }

  template <typename TYPE> struct Operation_Scalar {
    DEVICE INLINE void operator()(TYPE &out, TYPE &outF, TYPE &outG, TYPE &outH) {
      out = keops_ifelse(outF, outG, outH);
    }
  };

  template <class V, class GRADIN>
  using DiffT = IfElse<F, typename G::template DiffT<V, GRADIN>, typename H::template DiffT<V, GRADIN>>;
};

#define IfElse(f, g, h) KeopsNS<IfElse<decltype(InvKeopsNS(f)), decltype(InvKeopsNS(g)),decltype(InvKeopsNS(h))>>()

} // namespace keops
