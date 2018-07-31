#include "core/autodiff.h"
#include "core/CpuConv.cpp"

using namespace keops;

/////////////////////////
// Convolutions on Cpu //
/////////////////////////

// sum over j : gamma_i = sum_j F(X_i,Y_j)
extern "C" int CpuConv(int nx, int ny, __TYPE__* gamma, __TYPE__** args) {
    return SumReduction<F>::template Eval<CpuConv>(nx, ny, gamma, args);
}

// sum over i : gamma_j = sum_i F(X_i,Y_j)
extern "C" int CpuTransConv(int nx, int ny, __TYPE__* gamma, __TYPE__** args) {
    return SumReduction<F,1>::template Eval<CpuConv>(ny, nx, gamma, args);
}

