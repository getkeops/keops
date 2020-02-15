#include "core/reductions/Reduction.h"

using namespace keops;

///////////////////////////////////////////////
// Convolutions on Gpu device from host data //
///////////////////////////////////////////////

#include "core/mapreduce/GpuConv1D.cu"

extern "C" int GpuReduc1D_FromHost(int nx, int ny, __TYPE__ *gamma, __TYPE__ **args, int device_id = -1) {
  return Eval< F, GpuConv1D_FromHost >::Run(nx, ny, gamma, args, device_id);
}

#include "core/mapreduce/GpuConv2D.cu"

extern "C" int GpuReduc2D_FromHost(int nx, int ny, __TYPE__ *gamma, __TYPE__ **args, int device_id = -1) {
  return Eval< F, GpuConv2D_FromHost >::Run(nx, ny, gamma, args, device_id);
}

//////////////////////////////////////////////////////////
// Convolutions on Gpu device directly from device data //
//////////////////////////////////////////////////////////

extern "C" int GpuReduc1D_FromDevice(int nx, int ny, __TYPE__ *gamma, __TYPE__ **args, int device_id = -1) {
  return Eval< F, GpuConv1D_FromDevice >::Run(nx, ny, gamma, args, device_id);
}

extern "C" int GpuReduc2D_FromDevice(int nx, int ny, __TYPE__ *gamma, __TYPE__ **args, int device_id = -1) {
  return Eval< F, GpuConv2D_FromDevice >::Run(nx, ny, gamma, args, device_id);
}

/////////////////////////
// Convolutions on Cpu //
/////////////////////////

#include "core/mapreduce/CpuConv.cpp"

extern "C" int CpuReduc(int nx, int ny, __TYPE__ *gamma, __TYPE__ **args) {
  return Eval< F, CpuConv >::Run(nx, ny, gamma, args);
}


//////////////////////////////////////
// Convolutions on Cpu, with ranges //
//////////////////////////////////////

#include "core/mapreduce/CpuConv_ranges.cpp"

extern "C" int CpuReduc_ranges(int nx,
                               int ny,
                               int nbatchdims,
                               int *shapes,
                               int nranges_x,
                               int nranges_y,
                               __INDEX__ **castedranges,
                               __TYPE__ *gamma,
                               __TYPE__ **args) {
  return Eval< F, CpuConv_ranges >::Run(nx, ny, nbatchdims, shapes, nranges_x, nranges_y, castedranges, gamma, args);
}


//////////////////////////////////////
// Convolutions on GPU, with ranges //
//////////////////////////////////////

#include "core/mapreduce/GpuConv1D_ranges.cu"

extern "C" int GpuReduc1D_ranges_FromHost(int nx, int ny,
                                          int nbatchdims, int *shapes,
                                          int nranges_x, int nranges_y,
                                          int nredranges_x, int nredranges_y, __INDEX__ **castedranges,
                                          __TYPE__ *gamma, __TYPE__ **args, int device_id = -1) {
  return Eval< F, GpuConv1D_ranges_FromHost >::Run(nx,
                                                   ny,
                                                   nbatchdims,
                                                   shapes,
                                                   nranges_x,
                                                   nranges_y,
                                                   nredranges_x,
                                                   nredranges_y,
                                                   castedranges,
                                                   gamma,
                                                   args,
                                                   device_id);
}

extern "C" int GpuReduc1D_ranges_FromDevice(int nx, int ny,
                                            int nbatchdims, int *shapes,
                                            int nranges_x, int nranges_y, __INDEX__ **castedranges,
                                            __TYPE__ *gamma, __TYPE__ **args, int device_id = -1) {
  return Eval< F, GpuConv1D_ranges_FromDevice >::Run(nx,
                                                     ny,
                                                     nbatchdims,
                                                     shapes,
                                                     nranges_x,
                                                     nranges_y,
                                                     castedranges,
                                                     gamma,
                                                     args,
                                                     device_id);
}

