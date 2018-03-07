#include "core/autodiff.h"
#include "core/GpuConv1D.cu"
#include "core/GpuConv2D.cu"


///////////////////////////////////////////////
// Convolutions on Gpu device from host data //
///////////////////////////////////////////////

// sum over j : gamma_i = sum_j F(X_i,Y_j)
extern "C" int GpuConv2D(int nx, int ny, __TYPE__* gamma, __TYPE__** args) {
    return GpuConv2D(Generic<F>::sEval(), nx, ny, gamma, args);
}

// sum over i : gamma_j = sum_i F(X_i,Y_j)
extern "C" int GpuTransConv2D(int nx, int ny, __TYPE__* gamma, __TYPE__** args) {
    return GpuConv2D(Generic<F,1>::sEval(), ny, nx, gamma, args);
}

// sum over j : gamma_i = sum_j F(X_i,Y_j)
extern "C" int GpuConv1D(int nx, int ny, __TYPE__* gamma, __TYPE__** args) {
    return GpuConv1D(Generic<F>::sEval(), nx, ny, gamma, args);
}

// sum over i : gamma_j = sum_i F(X_i,Y_j)
extern "C" int GpuTransConv1D(int nx, int ny, __TYPE__* gamma, __TYPE__** args) {
    return GpuConv1D(Generic<F,1>::sEval(), ny, nx, gamma, args);
}

//////////////////////////////////////////////////////////
// Convolutions on Gpu device directly from device data //
//////////////////////////////////////////////////////////

// sum over j : gamma_i = sum_j F(X_i,Y_j)
extern "C" int GpuConv2D_FromDevice(int nx, int ny, __TYPE__* gamma, __TYPE__** args) {
    return GpuConv2D_FromDevice(Generic<F>::sEval(), nx, ny, gamma, args);
}

// sum over i : gamma_j = sum_i F(X_i,Y_j)
extern "C" int GpuTransConv2D_FromDevice(int nx, int ny, __TYPE__* gamma, __TYPE__** args) {
    return GpuConv2D_FromDevice(Generic<F,1>::sEval(), ny, nx, gamma, args);
}


// sum over j : gamma_i = sum_j F(X_i,Y_j)
extern "C" int GpuConv1D_FromDevice(int nx, int ny, __TYPE__* gamma, __TYPE__** args) {
    return GpuConv1D_FromDevice(Generic<F>::sEval(), nx, ny, gamma, args);
}

// sum over i : gamma_j = sum_i F(X_i,Y_j)
extern "C" int GpuTransConv1D_FromDevice(int nx, int ny, __TYPE__* gamma, __TYPE__** args) {
    return GpuConv1D_FromDevice(Generic<F,1>::sEval(), ny, nx, gamma, args);
}






/////////////////////////
// Convolutions on Cpu //
/////////////////////////

#include "core/CpuConv.cpp"

// sum over j : gamma_i = sum_j F(X_i,Y_j)
extern "C" int CpuConv(int nx, int ny, __TYPE__* gamma, __TYPE__** args) {
    return CpuConv(Generic<F>::sEval(), nx, ny, gamma, args);
}

// sum over i : gamma_j = sum_i F(X_i,Y_j)
extern "C" int CpuTransConv(int nx, int ny, __TYPE__* gamma, __TYPE__** args) {
    return CpuConv(Generic<F,1>::sEval(), ny, nx, gamma, args);
}



