#include "core/autodiff.h"
#include "core/GpuConv1D.cu"
#include "core/GpuConv2D.cu"
#include "core/reductions/sum.h"
#include "core/reductions/min.h"
#include "core/reductions/kmin.h"
#include "core/reductions/log_sum_exp.h"

using namespace keops;

///////////////////////////////////////////////
// Convolutions on Gpu device from host data //
///////////////////////////////////////////////

extern "C" int GpuReduc1D(int nx, int ny, __TYPE__* gamma, __TYPE__** args) {
    return Eval<F,GpuConv1D_FromHost>::Run(nx, ny, gamma, args);
}

extern "C" int GpuReduc2D(int nx, int ny, __TYPE__* gamma, __TYPE__** args) {
    return Eval<F,GpuConv2D_FromHost>::Run(nx, ny, gamma, args);
}

//////////////////////////////////////////////////////////
// Convolutions on Gpu device directly from device data //
//////////////////////////////////////////////////////////

extern "C" int GpuReduc1D_FromDevice(int nx, int ny, __TYPE__* gamma, __TYPE__** args) {
    return Eval<F,GpuConv1D_FromDevice>::Run(nx, ny, gamma, args);
}

extern "C" int GpuReduc2D_FromDevice(int nx, int ny, __TYPE__* gamma, __TYPE__** args) {
    return Eval<F,GpuConv2D_FromDevice>::Run(nx, ny, gamma, args);
}

/////////////////////////
// Convolutions on Cpu //
/////////////////////////

#include "core/CpuConv.cpp"

extern "C" int CpuReduc(int nx, int ny, __TYPE__* gamma, __TYPE__** args) {
    return Eval<F,CpuConv>::Run(nx, ny, gamma, args);
}

