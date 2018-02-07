#pragma once

#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <vector>

#include "Pack.h"
#include "reductions/sum.h"
#include "reductions/log_sum_exp.h"

using namespace std;

// Host implementation of the convolution, for comparison


template < typename TYPE, class FUN, class PARAM >
int CpuConv_(FUN fun, PARAM param, int nx, int ny, TYPE** px, TYPE** py) {
    typedef typename FUN::DIMSX DIMSX;
    typedef typename FUN::DIMSY DIMSY;
    const int DIMX = DIMSX::SUM;
    const int DIMY = DIMSY::SUM;
    const int DIMX1 = DIMSX::FIRST;

    TYPE xi[DIMX], yj[DIMY], tmp[DIMX1];
    for(int i=0; i<nx; i++) {
        load<DIMSX>(i,xi,px);
        InitializeOutput<TYPE,DIMX1,typename FUN::FORM>()(tmp);   // tmp = 0
        for(int j=0; j<ny; j++) {
            load<DIMSY>(j,yj,py);
            call<DIMSX,DIMSY>(fun,xi,yj,param);
            ReducePair<TYPE,DIMX1,typename FUN::FORM>()(tmp, xi); // tmp += xi
        }
        for(int k=0; k<DIMX1; k++)
            px[0][i*DIMX1+k] = tmp[k];
    }

    return 0;
}

// Wrapper with an user-friendly input format for px and py.
template < typename TYPE, class FUN, class PARAM, typename... Args >
int CpuConv(FUN fun, PARAM param, int nx, int ny, TYPE* x1, Args... args) {
    typedef typename FUN::VARSI VARSI;
    typedef typename FUN::VARSJ VARSJ;

    const int SIZEI = VARSI::SIZE+1;
    const int SIZEJ = VARSJ::SIZE;

    using DIMSX = GetDims<VARSI>;
    using DIMSY = GetDims<VARSJ>;

    using INDSI = GetInds<VARSI>;
    using INDSJ = GetInds<VARSJ>;

    TYPE *px[SIZEI];
    TYPE *py[SIZEJ];

    px[0] = x1;
    getlist<INDSI>(px+1,args...);
    getlist<INDSJ>(py,args...);

    return CpuConv_(fun,param,nx,ny,px,py);
}

// Idem, but with args given as an array of arrays, instead of an explicit list of arrays.
template < typename TYPE, class FUN, class PARAM >
int CpuConv(FUN fun, PARAM param, int nx, int ny, TYPE* x1, TYPE** args) {
    typedef typename FUN::VARSI VARSI;
    typedef typename FUN::VARSJ VARSJ;

    const int SIZEI = VARSI::SIZE+1;
    const int SIZEJ = VARSJ::SIZE;

    using DIMSX = GetDims<VARSI>;
    using DIMSY = GetDims<VARSJ>;

    using INDSI = GetInds<VARSI>;
    using INDSJ = GetInds<VARSJ>;

    TYPE *px[SIZEI];
    TYPE *py[SIZEJ];

    px[0] = x1;
    for(int i=1; i<SIZEI; i++)
        px[i] = args[INDSI::VAL(i-1)];
    for(int i=0; i<SIZEJ; i++)
        py[i] = args[INDSJ::VAL(i)];

    return CpuConv_(fun,param,nx,ny,px,py);
}
