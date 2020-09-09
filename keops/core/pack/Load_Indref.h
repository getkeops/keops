# pragma once

#include "core/pack/Pack.h"

namespace keops {

template < class DIMS, class INDS, class INDSREF >
struct load_offsets_indref_Impl {
  template < typename TYPE >
  HOST_DEVICE static void Eval(int i, TYPE *xi, TYPE **px, int *offsets) {}
};

template < class INDSREF, int FIRSTDIM, int... NEXTDIMS, int FIRSTIND, int... NEXTINDS >
struct load_offsets_indref_Impl < pack<FIRSTDIM,NEXTDIMS...>, pack<FIRSTIND,NEXTINDS...>, INDSREF > {
  using NEXTDIM = pack<NEXTDIMS...>;
  using NEXTIND = pack<NEXTINDS...>;
  template < typename TYPE >
  HOST_DEVICE static void Eval(int i, TYPE *xi, TYPE **px, int *offsets) {
    int true_i = offsets[IndVal<INDSREF,FIRSTIND>::value] + i;
    #pragma unroll
    for (int k = 0; k < FIRSTDIM; k++) {
      xi[k] = px[FIRSTIND][true_i * FIRSTDIM + k];
    }
    load_offsets_indref_Impl<NEXTDIM,NEXTIND,INDSREF>::Eval(i, xi + FIRSTDIM, px, offsets + 1); 
  }
};

template < class DIMS, class INDS, class INDSREF, typename TYPE >
HOST_DEVICE static void load_indref(int i, TYPE *xi, TYPE **px, int *offsets) {
  load_offsets_indref_Impl<DIMS,INDS,INDSREF>::Eval(i, xi, px, offsets); 
}






}
