#pragma once

#include <stdio.h>
#include <assert.h>
#include <vector>

#ifdef USE_OPENMP
#include <omp.h>
#endif

#include "core/utils/TypesUtils.h"
#include "core/pack/Pack.h"
#include "core/pack/Load.h"
#include "core/pack/Call.h"
#include "core/pack/GetInds.h"
#include "broadcast_batch_dimensions.h"

// Host implementation of the convolution, for comparison

namespace keops {

struct CpuConv_ranges {
  template< typename TYPE, class FUN >
  static int CpuConv_ranges_(FUN fun, int nx, int ny,
                             int nbatchdims, int* shapes,
                             int nranges_x, int nranges_y, __INDEX__** ranges,
                             TYPE *out, TYPE **args) {
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
    
    typedef typename FUN::VARSI VARSI;
    typedef typename FUN::VARSJ VARSJ;
    typedef typename FUN::VARSP VARSP;
    
    const int SIZEI = VARSI::SIZE;
    const int SIZEJ = VARSJ::SIZE;
    const int SIZEP = VARSP::SIZE;

    
  
    // Separate and store the shapes of the "i" and "j" variables + parameters --------------
    //
    // shapes is an array of size (1+nargs)*(nbatchdims+3), which looks like:
    // [ A, .., B, M, N, D_out]  -> output
    // [ A, .., B, M, 1, D_1  ]  -> "i" variable
    // [ A, .., B, 1, N, D_2  ]  -> "j" variable
    // [ A, .., B, 1, 1, D_3  ]  -> "parameter"
    // [ A, .., 1, M, 1, D_4  ]  -> N.B.: we support broadcasting on the batch dimensions!
    // [ 1, .., 1, M, 1, D_5  ]  ->      (we'll just ask users to fill in the shapes with *explicit* ones)
    
    int shapes_i[(SIZEI) * (nbatchdims + 1)], shapes_j[SIZEJ * (nbatchdims + 1)],
            shapes_p[SIZEP * (nbatchdims + 1)];
    
    // First, we fill shapes_i with the "relevant" shapes of the "i" variables,
    // making it look like, say:
    // [ A, .., B, M]
    // [ A, .., 1, M]
    // [ A, .., A, M]
    // Then, we do the same for shapes_j, but with "N" instead of "M".
    // And finally for the parameters, with "1" instead of "M".
    fill_shapes< FUN >(nbatchdims, shapes, shapes_i, shapes_j, shapes_p);
  
    // Actual for-for loop -----------------------------------------------------

    TYPE pp[DIMP];
    load< DIMSP, INDSP >(0, pp, args);  // If nbatchdims == 0, the parameters are fixed once and for all
        
    // Set the output to zero, as the ranges may not cover the full output -----
    __TYPEACC__ acctmp[DIMRED];
    for (int i = 0; i < nx; i++) {
      typename FUN::template InitializeReduction< __TYPEACC__, TYPE >()(acctmp);
      typename FUN::template FinalizeOutput< __TYPEACC__, TYPE >()(acctmp, out + i * DIMOUT, i);
    }
    
    // N.B.: In the following code, we assume that the x-ranges do not overlap.
    //       Otherwise, we'd have to assume that DIMRED == DIMOUT
    //       or allocate a buffer of size nx * DIMRED. This may be done in the future.
    // Cf. reduction.h: 
    //    FUN::tagJ = 1 for a reduction over j, result indexed by i
    //    FUN::tagJ = 0 for a reduction over i, result indexed by j
    
    int nranges = FUN::tagJ ? nranges_x : nranges_y;
    __INDEX__* ranges_x = FUN::tagJ ? ranges[0] : ranges[3];
    __INDEX__* slices_x = FUN::tagJ ? ranges[1] : ranges[4];
    __INDEX__* ranges_y = FUN::tagJ ? ranges[2] : ranges[5];

    int indices_i[SIZEI], indices_j[SIZEJ], indices_p[SIZEP];  // Buffers for the "broadcasted indices"
    for (int k = 0; k < SIZEI; k++) { indices_i[k] = 0; }  // Fill the "offsets" with zeroes,
    for (int k = 0; k < SIZEJ; k++) { indices_j[k] = 0; }  // the default value when nbatchdims == 0.
    for (int k = 0; k < SIZEP; k++) { indices_p[k] = 0; }
    
    for (int range_index = 0; range_index < nranges; range_index++) {

      __INDEX__ start_x = ranges_x[2 * range_index];
      __INDEX__ end_x = ranges_x[2 * range_index + 1];
  
      __INDEX__ start_slice = (range_index < 1) ? 0 : slices_x[range_index - 1];
      __INDEX__ end_slice = slices_x[range_index];
  
      // If needed, compute the "true" start indices of the range, turning
      // the "abstract" index start_x into an array of actual "pointers/offsets" stored in indices_i:
      if (nbatchdims > 0) {
        vect_broadcast_index(start_x, nbatchdims, SIZEI, shapes, shapes_i, indices_i);
        // And for the parameters, too:
        vect_broadcast_index(range_index, nbatchdims, SIZEP, shapes, shapes_p, indices_p);
        load< DIMSP, INDSP >(0, pp, args, indices_p); // Load the paramaters, once per tile
      }

#pragma omp parallel for   
      for (__INDEX__ i = start_x; i < end_x; i++) {
        TYPE xi[DIMX], yj[DIMY], fout[DIMFOUT];
        __TYPEACC__ acc[DIMRED];
#if SUM_SCHEME == BLOCK_SUM
        // additional tmp vector to store intermediate results from each block
        TYPE tmp[DIMRED];
#elif SUM_SCHEME == KAHAN_SCHEME
        // additional tmp vector to accumulate errors
        const int DIM_KAHAN = FUN::template KahanScheme<__TYPEACC__,TYPE>::DIMACC;
        TYPE tmp[DIM_KAHAN];
#endif
        if (nbatchdims == 0) {
          load< DIMSX, INDSI >(i, xi, args);
        } else {
          load< DIMSX, INDSI >(i - start_x, xi, args, indices_i);
        }
        typename FUN::template InitializeReduction< __TYPEACC__, TYPE >()(acc);   // tmp = 0
#if SUM_SCHEME == BLOCK_SUM
        typename FUN::template InitializeReduction< TYPE, TYPE >()(tmp);   // tmp = 0
#elif SUM_SCHEME == KAHAN_SCHEME
        VectAssign<DIM_KAHAN>(tmp,0.0f);
#endif
        for (__INDEX__ slice = start_slice; slice < end_slice; slice++) {
          __INDEX__ start_y = ranges_y[2 * slice];
          __INDEX__ end_y = ranges_y[2 * slice + 1];
      
          // If needed, compute the "true" start indices of the range, turning
          // the "abstract" index start_y into an array of actual "pointers/offsets" stored in indices_j:
          if (nbatchdims > 0) {
            vect_broadcast_index(start_y, nbatchdims, SIZEJ, shapes, shapes_j, indices_j);
          }

          if (nbatchdims == 0) {
            for (int j = start_y; j < end_y; j++) {
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
          }
          else {
            for (int j = start_y; j < end_y; j++) {
              load< DIMSY, INDSJ >(j - start_y, yj, args, indices_j);
              call< DIMSX, DIMSY, DIMSP >(fun, fout, xi, yj, pp);
#if SUM_SCHEME == BLOCK_SUM
              typename FUN::template ReducePairShort< TYPE, TYPE >()(tmp, fout, j - start_y); // tmp += fout
              if ((j+1)%200) {
                  typename FUN::template ReducePair< __TYPEACC__, TYPE >()(acc, tmp); // acc += tmp
                  typename FUN::template InitializeReduction< TYPE, TYPE >()(tmp);   // tmp = 0
              }
#elif SUM_SCHEME == KAHAN_SCHEME
              typename FUN::template KahanScheme<__TYPEACC__,TYPE>()(acc, fout, tmp);
#else
              typename FUN::template ReducePairShort< __TYPEACC__, TYPE >()(acc, fout, j - start_y); // acc += fout
#endif
            }
          }
        }
#if SUM_SCHEME == BLOCK_SUM
        typename FUN::template ReducePair< __TYPEACC__, TYPE >()(acc, tmp); // acc += tmp
#endif
        typename FUN::template FinalizeOutput< __TYPEACC__, TYPE >()(acc, out + i * DIMOUT, i);
      }

    }

    return 0;
  }

// Wrapper with an user-friendly input format for px and py.
  template< typename TYPE, class FUN, typename... Args >
  static int Eval(FUN fun, int nx, int ny,
                  int nbatchdims, int* shapes,
                  int nranges_x, int nranges_y, __INDEX__** ranges,
                  TYPE* out, Args... args) {
    static const int Nargs = sizeof...(Args);
    TYPE *pargs[Nargs];
    unpack(pargs, args...);
    return CpuConv_ranges_(fun, nx, ny, nbatchdims, shapes, nranges_x, nranges_y, ranges, out, pargs);
  }

// Idem, but with args given as an array of arrays, instead of an explicit list of arrays.
  template< typename TYPE, class FUN >
  static int Eval(FUN fun, int nx, int ny,
                  int nbatchdims, int* shapes,
                  int nranges_x, int nranges_y, __INDEX__** ranges,
                  TYPE* out, TYPE** pargs) {
    return CpuConv_ranges_(fun, nx, ny, nbatchdims, shapes, nranges_x, nranges_y, ranges, out, pargs);
  }
};
}
