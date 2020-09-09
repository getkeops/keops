#pragma once

#include <stdio.h>
#include <assert.h>
#include <vector>

#include "core/utils/TypesUtils.h"

#ifdef USE_OPENMP
#include <omp.h>
#endif

#include "core/pack/GetInds.h"
#include "core/pack/Load.h"
#include "core/pack/Call.h"

// Host implementation of the convolution, for comparison

namespace keops {

struct CpuConv {
  template < typename TYPE, class FUN >
  static int CpuConv_(FUN fun, int nx, int ny, TYPE *out, TYPE **args) {
    typedef typename FUN::DIMSX DIMSX; // dimensions of "i" indexed variables
    typedef typename FUN::DIMSY DIMSY; // dimensions of "j" indexed variables
    typedef typename FUN::DIMSP DIMSP; // dimensions of parameters variables
    typedef typename FUN::INDSI INDSI;
    typedef typename FUN::INDSJ INDSJ;
    typedef typename FUN::INDSP INDSP;
    const int DIMX = DIMSX::SUM; // total size of "i" indexed variables
    const int DIMY = DIMSY::SUM; // total size of "j" indexed variables
    const int DIMP = DIMSP::SUM; // total size of parameters variables
    const int DIMOUT = FUN::DIM; // dimension of output variable
    const int DIMRED = FUN::DIMRED; // dimension of reduction operation
    const int DIMFOUT = FUN::F::DIM; // dimension of output variable of inner function
    TYPE pp[DIMP];
    load< DIMSP, INDSP >(0, pp, args);

#pragma omp parallel for 
    for (int i = 0; i < nx; i++) {
    TYPE fout[DIMFOUT], xi[DIMX], yj[DIMY];
    __TYPEACC__ acc[DIMRED];
#if SUM_SCHEME == BLOCK_SUM
    // additional tmp vector to store intermediate results from each block
    TYPE tmp[DIMRED];
#elif SUM_SCHEME == KAHAN_SCHEME
    // additional tmp vector to accumulate errors
    const int DIM_KAHAN = FUN::template KahanScheme<__TYPEACC__,TYPE>::DIMACC;
    TYPE tmp[DIM_KAHAN];
#endif
      load< DIMSX, INDSI >(i, xi, args);
      typename FUN::template InitializeReduction< __TYPEACC__, TYPE >()(acc);   // acc = 0
#if SUM_SCHEME == BLOCK_SUM
      typename FUN::template InitializeReduction< TYPE, TYPE >()(tmp);   // tmp = 0
#elif SUM_SCHEME == KAHAN_SCHEME
      VectAssign<DIM_KAHAN>(tmp,0.0f);
#endif
      for (int j = 0; j < ny; j++) {
        load< DIMSY, INDSJ >(j, yj, args);
        call< DIMSX, DIMSY, DIMSP >(fun, fout, xi, yj, pp);
#if SUM_SCHEME == BLOCK_SUM
        typename FUN::template ReducePairShort< TYPE, TYPE >()(tmp, fout, j); // tmp += fout
        if ((j+1)%200) {
            typename FUN::template ReducePair< __TYPEACC__, TYPE >()(acc, tmp); // acc += tmp
            typename FUN::template InitializeReduction< TYPE, TYPE >()(tmp);   // tmp = 0
        }
#elif SUM_SCHEME == KAHAN_SCHEME
        typename FUN::template KahanScheme<__TYPEACC__,TYPE>()(acc, fout, tmp);
#else
        typename FUN::template ReducePairShort< __TYPEACC__, TYPE >()(acc, fout, j); // acc += fout
#endif
      }
#if SUM_SCHEME == BLOCK_SUM
      typename FUN::template ReducePair< __TYPEACC__, TYPE >()(acc, tmp); // acc += tmp
#endif          
      typename FUN::template FinalizeOutput< __TYPEACC__, TYPE >()(acc, out + i * DIMOUT, i);
    }

    return 0;
  }

// Wrapper with an user-friendly input format for px and py.
  template < typename TYPE, class FUN, typename... Args >
  static int Eval(FUN fun, int nx, int ny, TYPE *out, Args... args) {
    static const int Nargs = sizeof...(Args);
    TYPE *pargs[Nargs];
    unpack(pargs, args...);
    return CpuConv_(fun, nx, ny, out, pargs);
  }

// Idem, but with args given as an array of arrays, instead of an explicit list of arrays.
  template < typename TYPE, class FUN >
  static int Eval(FUN fun, int nx, int ny, TYPE *out, TYPE **pargs) {
    return CpuConv_(fun, nx, ny, out, pargs);
  }
};
}
