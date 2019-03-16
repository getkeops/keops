#pragma once

#include <stdio.h>
#include <assert.h>
#include <vector>

#include "core/Pack.h"

// Host implementation of the convolution, for comparison

namespace keops {

struct CpuConv_ranges {
template < typename TYPE, class FUN >
static int CpuConv_ranges_(FUN fun, TYPE** param, int nx, int ny, 
                           int nranges_x, int nranges_y, __INDEX__ **ranges, 
                           TYPE** px, TYPE** py) {
    typedef typename FUN::DIMSX DIMSX; // dimensions of "i" indexed variables
    typedef typename FUN::DIMSY DIMSY; // dimensions of "j" indexed variables
    typedef typename FUN::DIMSP DIMSP; // dimensions of parameters variables
    const int DIMX = DIMSX::SUM; // total size of "i" indexed variables
    const int DIMY = DIMSY::SUM; // total size of "j" indexed variables
    const int DIMP = DIMSP::SUM; // total size of parameters variables
    const int DIMOUT = FUN::DIM; // dimension of output variable
    const int DIMRED = FUN::DIMRED; // dimension of reduction operation
    const int DIMFOUT = DIMSX::FIRST; // dimension of output variable of inner function
    TYPE xi[DIMX], yj[DIMY], pp[DIMP], tmp[DIMRED];
    load<DIMSP>(0,pp,param);


    // Set the output to zero, as the ranges may not cover the full output
    for(int i=0; i<nx; i++) {
        typename FUN::template InitializeReduction<TYPE>()(tmp); 
        typename FUN::template FinalizeOutput<TYPE>()(tmp, px[0]+i*DIMOUT, px, i);
    }

    // N.B.: In the following code, we assume that the x-ranges do not overlap.
    //       Otherwise, we'd have to assume that DIMRED == DIMOUT
    //       or allocate a buffer of size nx * DIMRED. This may be done in the future.
    // Cf. reduction.h: 
    //    FUN::tagJ = 1 for a reduction over j, result indexed by i
    //    FUN::tagJ = 0 for a reduction over i, result indexed by j

    int nranges = FUN::tagJ ? nranges_x : nranges_y ;
    __INDEX__ *ranges_x = FUN::tagJ ? ranges[0] : ranges[3] ;
    __INDEX__ *slices_x = FUN::tagJ ? ranges[1] : ranges[4] ;
    __INDEX__ *ranges_y = FUN::tagJ ? ranges[2] : ranges[5] ;

    for(int range_index = 0; range_index < nranges ; range_index++) {
        __INDEX__ start_x = ranges_x[2*range_index] ;
        __INDEX__ end_x   = ranges_x[2*range_index + 1] ;

        __INDEX__ start_slice = (range_index < 1) ? 0 : slices_x[range_index-1] ;
        __INDEX__ end_slice   = slices_x[range_index] ;

        for(__INDEX__ i = start_x; i < end_x; i++) {
            load<typename DIMSX::NEXT>(i,xi+DIMFOUT,px+1);
            typename FUN::template InitializeReduction<TYPE>()(tmp);   // tmp = 0

            for(__INDEX__ slice = start_slice ; slice < end_slice ; slice++) { 
                __INDEX__ start_y = ranges_y[2*slice] ;
                __INDEX__ end_y   = ranges_y[2*slice + 1] ;

                for(int j = start_y; j < end_y; j++) {
                    load<DIMSY>(j,yj,py);
                    call<DIMSX,DIMSY,DIMSP>(fun,xi,yj,pp);
                    typename FUN::template ReducePairShort<TYPE>()(tmp, xi, j); // tmp += xi
                }
            }
            typename FUN::template FinalizeOutput<TYPE>()(tmp, px[0]+i*DIMOUT, px, i);
        }

    }

    return 0;
}

// Wrapper with an user-friendly input format for px and py.
template < typename TYPE, class FUN, typename... Args >
static int Eval(FUN fun, int nx, int ny, 
                int nranges_x, int nranges_y, __INDEX__ **ranges,
                TYPE* x1, Args... args) {
    typedef typename FUN::VARSI VARSI;
    typedef typename FUN::VARSJ VARSJ;
    typedef typename FUN::VARSP VARSP;

    const int SIZEI = VARSI::SIZE+1;
    const int SIZEJ = VARSJ::SIZE;
    const int SIZEP = VARSP::SIZE;

    using INDSI = GetInds<VARSI>;
    using INDSJ = GetInds<VARSJ>;
    using INDSP = GetInds<VARSP>;

    TYPE *px[SIZEI];
    TYPE *py[SIZEJ];
    TYPE *params[SIZEP];
    px[0] = x1;
    getlist<INDSI>(px+1,args...);
    getlist<INDSJ>(py,args...);
    getlist<INDSP>(params,args...);

    return CpuConv_ranges_(fun,params,nx,ny,nranges_x,nranges_y,ranges,px,py);
}

// Idem, but with args given as an array of arrays, instead of an explicit list of arrays.
template < typename TYPE, class FUN >
static int Eval(FUN fun, int nx, int ny, 
                int nranges_x, int nranges_y, __INDEX__ **ranges,
                TYPE* x1, TYPE** args) {
    typedef typename FUN::VARSI VARSI;
    typedef typename FUN::VARSJ VARSJ;
    typedef typename FUN::VARSP VARSP;

    const int SIZEI = VARSI::SIZE+1;
    const int SIZEJ = VARSJ::SIZE;
    const int SIZEP = VARSP::SIZE;

    using INDSI = GetInds<VARSI>;
    using INDSJ = GetInds<VARSJ>;
    using INDSP = GetInds<VARSP>;

    TYPE *px[SIZEI];
    TYPE *py[SIZEJ];
    TYPE *params[SIZEP];

    px[0] = x1;
    for(int i=1; i<SIZEI; i++)
        px[i] = args[INDSI::VAL(i-1)];
    for(int i=0; i<SIZEJ; i++)
        py[i] = args[INDSJ::VAL(i)];
    for(int i=0; i<SIZEP; i++)
        params[i] = args[INDSP::VAL(i)];
    return CpuConv_ranges_(fun,params,nx,ny,nranges_x,nranges_y,ranges,px,py);
}
};
}
