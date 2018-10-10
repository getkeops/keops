#pragma once

#include <stdio.h>
#include <assert.h>
#include <vector>

#include "core/Pack.h"

// Host implementation of the convolution, for comparison

namespace keops {

struct CpuConv {
template < typename TYPE, class FUN >
static int CpuConv_(FUN fun, TYPE** param, int nx, int ny, TYPE** px, TYPE** py) {
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

    for(int i=0; i<nx; i++) {
        load<typename DIMSX::NEXT>(i,xi+DIMFOUT,px+1);
        typename FUN::template InitializeReduction<TYPE>()(tmp);   // tmp = 0
        for(int j=0; j<ny; j++) {
            load<DIMSY>(j,yj,py);
            call<DIMSX,DIMSY,DIMSP>(fun,xi,yj,pp);
            typename FUN::template ReducePairShort<TYPE>()(tmp, xi, j); // tmp += xi
        }
        typename FUN::template FinalizeOutput<TYPE>()(tmp, px[0]+i*DIMOUT, px, i);
    }

    return 0;
}

// Wrapper with an user-friendly input format for px and py.
template < typename TYPE, class FUN, typename... Args >
static int Eval(FUN fun, int nx, int ny, TYPE* x1, Args... args) {
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

    return CpuConv_(fun,params,nx,ny,px,py);
}

// Idem, but with args given as an array of arrays, instead of an explicit list of arrays.
template < typename TYPE, class FUN >
static int Eval(FUN fun, int nx, int ny, TYPE* x1, TYPE** args) {
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
    return CpuConv_(fun,params,nx,ny,px,py);
}
};
}
