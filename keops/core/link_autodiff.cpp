#include "core/reductions/Reduction.h"

using namespace keops;

#if !USE_HALF

extern "C" int GetFormulaConstants(int *out) {
	
    constexpr int POS = std::max(F::POS_FIRST_ARGI, F::POS_FIRST_ARGJ);
    static_assert(((POS > -1) || (F::NVARS > 0)) , "[KeOps] There is no variables detected in the formula.");
	
	out[0] = F::NMINARGS;
	out[1] = F::tagI;
	out[2] = F::POS_FIRST_ARGI;
	out[3] = F::POS_FIRST_ARGJ;
	out[4] = F::NVARS;
	out[5] = F::NVARSI;
	out[6] = F::NVARSJ;
	out[7] = F::NVARSP;
	out[8] = F::DIM;
	return 0;
}

extern "C" int GetIndsI(int *out) {
	for (int k=0; k<F::NVARSI; k++)
		out[k] = F::INDSI::VAL(k);
	return 0;
}

extern "C" int GetIndsJ(int *out) {
	for (int k=0; k<F::NVARSJ; k++)
		out[k] = F::INDSJ::VAL(k);
	return 0;
}

extern "C" int GetIndsP(int *out) {
	for (int k=0; k<F::NVARSP; k++)
		out[k] = F::INDSP::VAL(k);
	return 0;
}

extern "C" int GetDimsX(int *out) {
	for (int k=0; k<F::NVARSI; k++)
		out[k] = F::DIMSX::VAL(k);
	return 0;
}

extern "C" int GetDimsY(int *out) {
	for (int k=0; k<F::NVARSJ; k++)
		out[k] = F::DIMSY::VAL(k);
	return 0;
}

extern "C" int GetDimsP(int *out) {
	for (int k=0; k<F::NVARSP; k++)
		out[k] = F::DIMSP::VAL(k);
	return 0;
}



/////////////////////////
// Convolutions on Cpu //
/////////////////////////

#include "core/mapreduce/CpuConv.cpp"

extern "C" int CpuReduc(int nx, int ny, __TYPE__* gamma, __TYPE__** args) {
  return Eval< F, CpuConv >::Run(nx, ny, gamma, args);
}


//////////////////////////////////////
// Convolutions on Cpu, with ranges //
//////////////////////////////////////

#include "core/mapreduce/CpuConv_ranges.cpp"

extern "C" int CpuReduc_ranges(int nx, int ny, int nbatchdims, int *shapes, int nranges_x, int nranges_y, __INDEX__ **castedranges, __TYPE__* gamma, __TYPE__** args) {
  return Eval< F, CpuConv_ranges >::Run(nx, ny, nbatchdims, shapes, nranges_x, nranges_y, castedranges, gamma, args);
}

#endif
