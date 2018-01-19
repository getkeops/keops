
#ifdef USENEWSYNTAX
	#include "core/newsyntax.h"
	using F = decltype(FORMULA);
#else
	using F = FORMULA;
#endif

///////////////////////////////////////////////
// Convolutions on Gpu device from host data //
///////////////////////////////////////////////

// sum over j : gamma_i = sum_j F(X_i,Y_j)
extern "C" int GpuConv2D(__TYPE__* params, int nx, int ny, __TYPE__* gamma, __TYPE__** args) {
    return GpuConv2D(Generic<F>::sEval(), params, nx, ny, gamma, args);
}

// sum over i : gamma_j = sum_i F(X_i,Y_j)
extern "C" int GpuTransConv2D(__TYPE__* params, int nx, int ny, __TYPE__* gamma, __TYPE__** args) {
    return GpuConv2D(Generic<F,1>::sEval(), params, ny, nx, gamma, args);
}

// sum over j : gamma_i = sum_j F(X_i,Y_j)
extern "C" int GpuConv1D(__TYPE__* params, int nx, int ny, __TYPE__* gamma, __TYPE__** args) {
    return GpuConv1D(Generic<F>::sEval(), params, nx, ny, gamma, args);
}

// sum over i : gamma_j = sum_i F(X_i,Y_j)
extern "C" int GpuTransConv1D(__TYPE__* params, int nx, int ny, __TYPE__* gamma, __TYPE__** args) {
    return GpuConv1D(Generic<F,1>::sEval(), params, ny, nx, gamma, args);
}

//////////////////////////////////////////////////////////
// Convolutions on Gpu device directly from device data //
//////////////////////////////////////////////////////////

// sum over j : gamma_i = sum_j F(X_i,Y_j)
extern "C" int GpuConv2D_FromDevice(__TYPE__* params, int nx, int ny, __TYPE__* gamma, __TYPE__** args) {
    return GpuConv2D_FromDevice(Generic<F>::sEval(), params, nx, ny, gamma, args);
}

// sum over i : gamma_j = sum_i F(X_i,Y_j)
extern "C" int GpuTransConv2D_FromDevice(__TYPE__* params, int nx, int ny, __TYPE__* gamma, __TYPE__** args) {
    return GpuConv2D_FromDevice(Generic<F,1>::sEval(), params, ny, nx, gamma, args);
}


// sum over j : gamma_i = sum_j F(X_i,Y_j)
extern "C" int GpuConv1D_FromDevice(__TYPE__* params, int nx, int ny, __TYPE__* gamma, __TYPE__** args) {
    return GpuConv1D_FromDevice(Generic<F>::sEval(), params, nx, ny, gamma, args);
}

// sum over i : gamma_j = sum_i F(X_i,Y_j)
extern "C" int GpuTransConv1D_FromDevice(__TYPE__* params, int nx, int ny, __TYPE__* gamma, __TYPE__** args) {
    return GpuConv1D_FromDevice(Generic<F,1>::sEval(), params, ny, nx, gamma, args);
}






/////////////////////////
// Convolutions on Cpu //
/////////////////////////

#include "core/CpuConv.cpp"

// sum over j : gamma_i = sum_j F(X_i,Y_j)
extern "C" int CpuConv(__TYPE__* params, int nx, int ny, __TYPE__* gamma, __TYPE__** args) {
    return CpuConv(Generic<F>::sEval(), params, nx, ny, gamma, args);
}

// sum over i : gamma_j = sum_i F(X_i,Y_j)
extern "C" int CpuTransConv(__TYPE__* params, int nx, int ny, __TYPE__* gamma, __TYPE__** args) {
    return CpuConv(Generic<F,1>::sEval(), params, ny, nx, gamma, args);
}


