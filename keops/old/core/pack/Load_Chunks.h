# pragma once

#include "core/pack/Pack.h"
#include "core/pack/IndVal.h"

namespace keops {


// load : loads chunks of variables, unrolling dimensions and indices.
//
// Signature is :
//   load_chunks< INDS, DIM_CHUNK, DIM_CHUNK_LOAD, DIM_ORG >
//		(int i, int chunk_index, TYPE *xi, TYPE **px);
//
// Example:
//   load_chunks< pack<7,9,8>, 3, 2, 11 >(5,k,xi,px);
// means : there are 3 chunks of vectors to load. They are located
// at positions 7, 8 and 9 in px. Now i=5 and DIM_ORG=11, so we start
// to load vectors at positions px[7]+5*11, px[8]+5*11, px[9]+5*11.
// For each, we load the kth chunk, assuming vector is divided
// into chunks of size 3. And finally, we stop after loading 2 values.
// 
// So we will execute:
//   xi[0] = px[7][5*11+k*3];
//   xi[1] = px[7][5*11+k*3+1];
//   xi[3] = px[9][5*11+k*3];
//   xi[4] = px[9][5*11+k*3+1];
//   xi[5] = px[8][5*11+k*3];
//   xi[6] = px[8][5*11+k*3+1];
  
template < class INDS, int DIM_CHUNK, int DIM_CHUNK_LOAD, int DIM_ORG >
struct load_chunks_Impl {
  template < typename TYPE >
  HOST_DEVICE static void Eval(int i, int chunk_index, TYPE *xi, TYPE **px) {}
};

template < int DIM_CHUNK, int DIM_CHUNK_LOAD, int DIM_ORG, int FIRSTIND, int... NEXTINDS >
struct load_chunks_Impl < pack<FIRSTIND,NEXTINDS...>, DIM_CHUNK, DIM_CHUNK_LOAD, DIM_ORG > {
  using NEXTIND = pack<NEXTINDS...>;
  template < typename TYPE >
  HOST_DEVICE static void Eval(int i, int chunk_index, TYPE *xi, TYPE **px) {
    #pragma unroll
    for (int k = 0; k < DIM_CHUNK_LOAD; k++) {
      xi[k] = px[FIRSTIND][i * DIM_ORG + chunk_index*DIM_CHUNK + k];
    }
    load_chunks_Impl<NEXTIND,DIM_CHUNK,DIM_CHUNK_LOAD,DIM_ORG>::Eval(i, chunk_index, xi + DIM_CHUNK, px); 
  }
};

template < class INDS, int DIM_CHUNK, int DIM_CHUNK_LOAD, int DIM_ORG, typename TYPE >
HOST_DEVICE static void load_chunks(int i, int chunk_index, TYPE *xi, TYPE **px) {
  load_chunks_Impl<INDS,DIM_CHUNK,DIM_CHUNK_LOAD,DIM_ORG>::Eval(i, chunk_index, xi, px); 
}


// Version with variable-dependent offsets (used when broadcasting batch dimensions)
// Example:
//   load_chunks< pack<2,3,1>, pack<8,9,7,3,1,2>, 3, 2, 11 >(5,k,xi,px);
// will execute:
//   xi[0] = px[2][(5+offsets[5])*11+k*3];
//   xi[1] = px[2][(5+offsets[5])*11+k*3+1];
//   xi[3] = px[3][(5+offsets[3])*11+k*3];
//   xi[4] = px[3][(5+offsets[3])*11+k*3+1];
//   xi[5] = px[1][(5+offsets[4])*11+k*3];
//   xi[6] = px[1][(5+offsets[4])*11+k*3+1];

template < class INDS, class INDSREF, int DIM_CHUNK, int DIM_CHUNK_LOAD, int DIM_ORG >
struct load_chunks_offsets_Impl {
  template < typename TYPE >
  HOST_DEVICE static void Eval(int i, int chunk_index, TYPE *xi, TYPE **px, int *offsets) {}
};

template < class INDSREF, int DIM_CHUNK, int DIM_CHUNK_LOAD, int DIM_ORG, int FIRSTIND, int... NEXTINDS >
struct load_chunks_offsets_Impl < pack<FIRSTIND,NEXTINDS...>, INDSREF, DIM_CHUNK, DIM_CHUNK_LOAD, DIM_ORG > {
  using NEXTIND = pack<NEXTINDS...>;
  template < typename TYPE >
  HOST_DEVICE static void Eval(int i, int chunk_index, TYPE *xi, TYPE **px, int *offsets) {
	int true_i = offsets[IndVal<INDSREF,FIRSTIND>::value] + i;
    #pragma unroll
    for (int k = 0; k < DIM_CHUNK_LOAD; k++)
      xi[k] = px[FIRSTIND][true_i * DIM_ORG + chunk_index*DIM_CHUNK + k];
    load_chunks_offsets_Impl<NEXTIND,INDSREF,DIM_CHUNK,DIM_CHUNK_LOAD,DIM_ORG>::Eval(i, chunk_index, xi + DIM_CHUNK, px, offsets); 
  }
};

template < class INDS, class INDSREF, int DIM_CHUNK, int DIM_CHUNK_LOAD, int DIM_ORG, typename TYPE >
HOST_DEVICE static void load_chunks(int i, int chunk_index, TYPE *xi, TYPE **px, int *offsets) {
  load_chunks_offsets_Impl<INDS,INDSREF,DIM_CHUNK,DIM_CHUNK_LOAD,DIM_ORG>::Eval(i, chunk_index, xi, px, offsets); 
}




}
