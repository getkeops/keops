#pragma once

#include "core/Pack.h"

#include "core/autodiff.h"

#include "core/reductions/reduction.h"

namespace keops {

// Implements the zero reduction operation (fills output with zeros)
// tagI is equal:
// - to 0 if you do the summation over j (with i the index of the output vector),
// - to 1 if you do the summation over i (with j the index of the output vector).
//
template < int DIM_, int tagI=0 >
struct Zero_Reduction : public Reduction<Zero<DIM_>,tagI> {

    static const int DIM = DIM_;	

    static void PrintId(std::stringstream& str) {
        str << "Zero_Reduction(DIM=" << DIM << ",tagI=" << tagI << ")";
    }

    template < class V, class GRADIN, class FO=void >
    using DiffT = Zero_Reduction<V::DIM,(V::CAT)%2>;
    // remark : if V::CAT is 2 (parameter), we will get tagI=(V::CAT)%2=0, so we will do reduction wrt j.
    // In this case there is a summation left to be done by the user.

};

// specialized evaluation : no need to call a reduction operation for filling zeros

template < int DIM, int tagI, class MODE >
struct Eval<Zero_Reduction<DIM,tagI>,MODE> {
    template < typename TYPE, typename... Args >
    static int Run(int nx, int ny, TYPE *out, Args... args) {
        for(int k=0; k<(tagI==0?nx:ny)*DIM; k++)
            out[k] = 0;
        return 0;
    }
};

// The signature of *Conv*_ranges is slightly different...

struct CpuConv_ranges;
struct GpuConv1D_ranges_FromHost;

template < int DIM, int tagI >
struct Eval<Zero_Reduction<DIM,tagI>,CpuConv_ranges> {
    template < typename TYPE, typename... Args >
    static int Run(int nx, int ny, 
                int nranges_x, int nranges_y, __INDEX__ **ranges,
                TYPE *out, Args... args) {
        for(int k=0; k<(tagI==0?nx:ny)*DIM; k++)
            out[k] = 0;
        return 0;
    }
};

template < int DIM, int tagI >
struct Eval<Zero_Reduction<DIM,tagI>,GpuConv1D_ranges_FromHost> {
    template < typename TYPE, typename... Args >
    static int Run(int nx, int ny, 
                int nranges_x, int nranges_y, int nredranges_x, int nredranges_y, __INDEX__ **ranges,
                TYPE *out, Args... args) {
        for(int k=0; k<(tagI==0?nx:ny)*DIM; k++)
            out[k] = 0;
        return 0;
    }
};


#ifdef __CUDACC__
// specializations in case of device data
struct GpuConv1D_FromDevice;
struct GpuConv2D_FromDevice;

template < int DIM, int tagI >
struct Eval<Zero_Reduction<DIM,tagI>,GpuConv1D_FromDevice> {
    template < typename TYPE, typename... Args >
    static int Run(int nx, int ny, TYPE *out, Args... args) {
        cudaMemset(out, 0, (tagI==0?nx:ny)*DIM*sizeof(TYPE));
        return 0;
    }
};

template < int DIM, int tagI >
struct Eval<Zero_Reduction<DIM,tagI>,GpuConv2D_FromDevice> {
    template < typename TYPE, typename... Args >
    static int Run(int nx, int ny, TYPE *out, Args... args) {
        cudaMemset(out, 0, (tagI==0?nx:ny)*DIM*sizeof(TYPE));
        return 0;
    }
};


// The signature of *Conv*_ranges is slightly different...

struct GpuConv1D_ranges_FromDevice;

template < int DIM, int tagI >
struct Eval<Zero_Reduction<DIM,tagI>,GpuConv1D_ranges_FromDevice> {
    template < typename TYPE, typename... Args >
    static int Run(int nx, int ny, 
                int nranges_x, int nranges_y, __INDEX__ **ranges,
                TYPE *out, Args... args) {
        cudaMemset(out, 0, (tagI==0?nx:ny)*DIM*sizeof(TYPE));
        return 0;
    }
};

#endif

}
