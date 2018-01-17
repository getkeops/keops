#include <iostream>

#include "core/autodiff.h"

#include <stdarg.h>
#include <stdio.h>

#ifdef USENEWSYNTAX
	#include "core/newsyntax.h"
	using F = decltype(FORMULA);
#else
	using F = FORMULA;
#endif







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



