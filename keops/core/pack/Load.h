# pragma once

#include "core/pack/Pack.h"

namespace keops {


// load : loads variables, unrolling dimensions and indices
// Example:
//   load< pack<2,2,3>, pack<7,9,8> >(5,xi,px);
// will execute:
//   xi[0] = px[7][5*2];
//   xi[1] = px[7][5*2+1];
//   xi[3] = px[9][5*2];
//   xi[4] = px[9][5*2+1];
//   xi[5] = px[8][5*3];
//   xi[6] = px[8][5*3+1];
//   xi[7] = px[8][5*3+2];
  
template < class DIMS, class INDS >
struct load_Impl {
  template < typename TYPE >
  HOST_DEVICE static void Eval(int i, TYPE *xi, TYPE **px) {}
};

template < int FIRSTDIM, int... NEXTDIMS, int FIRSTIND, int... NEXTINDS >
struct load_Impl < pack<FIRSTDIM,NEXTDIMS...>, pack<FIRSTIND,NEXTINDS...> > {
  using NEXTDIM = pack<NEXTDIMS...>;
  using NEXTIND = pack<NEXTINDS...>;
  template < typename TYPE >
  HOST_DEVICE static void Eval(int i, TYPE *xi, TYPE **px) {
    #pragma unroll
    for (int k = 0; k < FIRSTDIM; k++) {
      xi[k] = px[FIRSTIND][i * FIRSTDIM + k];
    }
    load_Impl<NEXTDIM,NEXTIND>::Eval(i, xi + FIRSTDIM, px); 
  }
};

template < class DIMS, class INDS, typename TYPE >
HOST_DEVICE static void load(int i, TYPE *xi, TYPE **px) {
  load_Impl<DIMS,INDS>::Eval(i, xi, px); 
}


// Version with variable-dependent offsets (used when broadcasting batch dimensions)
// Example:
//   load< pack<2,2,3>, pack<7,9,8> >(5,xi,px,offsets);
// will execute:
//   xi[0] = px[7][(5+offsets[0])*2];
//   xi[1] = px[7][(5+offsets[0])*2+1];
//   xi[3] = px[9][(5+offsets[1])*2];
//   xi[4] = px[9][(5+offsets[1])*2+1];
//   xi[5] = px[8][(5+offsets[2])*3];
//   xi[6] = px[8][(5+offsets[2])*3+1];
//   xi[7] = px[8][(5+offsets[2])*3+2];

template < class DIMS, class INDS >
struct load_offsets_Impl {
  template < typename TYPE >
  HOST_DEVICE static void Eval(int i, TYPE *xi, TYPE **px, int *offsets) {}
};

template < int FIRSTDIM, int... NEXTDIMS, int FIRSTIND, int... NEXTINDS >
struct load_offsets_Impl < pack<FIRSTDIM,NEXTDIMS...>, pack<FIRSTIND,NEXTINDS...> > {
  using NEXTDIM = pack<NEXTDIMS...>;
  using NEXTIND = pack<NEXTINDS...>;
  template < typename TYPE >
  HOST_DEVICE static void Eval(int i, TYPE *xi, TYPE **px, int *offsets) {
    int true_i = offsets[0] + i;
    #pragma unroll
    for (int k = 0; k < FIRSTDIM; k++) {
      xi[k] = px[FIRSTIND][true_i * FIRSTDIM + k];
    }
    load_offsets_Impl<NEXTDIM,NEXTIND>::Eval(i, xi + FIRSTDIM, px, offsets + 1); 
  }
};

template < class DIMS, class INDS, typename TYPE >
HOST_DEVICE static void load(int i, TYPE *xi, TYPE **px, int *offsets) {
  load_offsets_Impl<DIMS,INDS>::Eval(i, xi, px, offsets); 
}






}
