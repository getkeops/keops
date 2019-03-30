#include "core/autodiff.h"
#include "core/CpuConv.cpp"
#include "core/reductions/sum.h"
#include "core/reductions/min.h"
#include "core/reductions/kmin.h"
#include "core/reductions/max_sumshiftexp.h"

#include "core/formulas/newsyntax.h"

using namespace keops;

/////////////////////////
// Convolutions on Cpu //
/////////////////////////

extern "C" int CpuReduc(int nx, int ny, __TYPE__* gamma, __TYPE__** args) {
    return Eval<F,CpuConv>::Run(nx, ny, gamma, args);
}


//////////////////////////////////////
// Convolutions on Cpu, with ranges //
//////////////////////////////////////

#include "core/CpuConv_ranges.cpp"

extern "C" int CpuReduc_ranges(int nx, int ny, int nranges_x, int nranges_y, __INDEX__ **castedranges, __TYPE__* gamma, __TYPE__** args) {
    return Eval<F,CpuConv_ranges>::Run(nx, ny, nranges_x, nranges_y, castedranges, gamma, args);
}

