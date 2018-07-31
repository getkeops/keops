#include "core/autodiff.h"
#include "core/CpuConv.cpp"
#include "core/reductions/sum.h"
#include "core/reductions/min.h"
#include "core/reductions/kmin.h"
#include "core/reductions/log_sum_exp.h"

#include "core/formulas/newsyntax.h"

using namespace keops;

/////////////////////////
// Convolutions on Cpu //
/////////////////////////

extern "C" int CpuReduc(int nx, int ny, __TYPE__* gamma, __TYPE__** args) {
    return Eval<F,CpuConv>::Run(nx, ny, gamma, args);
}

