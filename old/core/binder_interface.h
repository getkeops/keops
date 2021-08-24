
using namespace keops;

// Interface with binders : we define functions to communicate properties of the formula to the binders

extern "C" int GetFormulaString(std::string &out) {
	out = PrintReduction<F>();
	return 0;
}
	
extern "C" int GetFormulaConstants(int *out) {
	
	out[0] = F::NMINARGS;
	out[1] = F::tagI;
	if(F::tagI==0) {
		out[2] = F::POS_FIRST_ARGI;
		out[3] = F::POS_FIRST_ARGJ;
		out[4] = F::NVARSI;
		out[5] = F::NVARSJ;
	}
	else {
		out[2] = F::POS_FIRST_ARGJ;
		out[3] = F::POS_FIRST_ARGI;
		out[4] = F::NVARSJ;
		out[5] = F::NVARSI;
	}
	out[6] = F::NVARSP;
	out[7] = F::DIM;
	return 0;
}

extern "C" int GetIndsI(int *out) {
	if(F::tagI==0)
		for (int k=0; k<F::NVARSI; k++)
			out[k] = F::INDSI::VAL(k);
	else
		for (int k=0; k<F::NVARSJ; k++)
			out[k] = F::INDSJ::VAL(k);
	return 0;
}

extern "C" int GetIndsJ(int *out) {
	if(F::tagI==0)
		for (int k=0; k<F::NVARSJ; k++)
			out[k] = F::INDSJ::VAL(k);
	else
		for (int k=0; k<F::NVARSI; k++)
			out[k] = F::INDSI::VAL(k);
	return 0;
}

extern "C" int GetIndsP(int *out) {
	for (int k=0; k<F::NVARSP; k++)
		out[k] = F::INDSP::VAL(k);
	return 0;
}

extern "C" int GetDimsX(int *out) {
	if(F::tagI==0)
		for (int k=0; k<F::NVARSI; k++)
			out[k] = F::DIMSX::VAL(k);
	else
		for (int k=0; k<F::NVARSJ; k++)
			out[k] = F::DIMSY::VAL(k);
	return 0;
}

extern "C" int GetDimsY(int *out) {
	if(F::tagI==0)
		for (int k=0; k<F::NVARSJ; k++)
			out[k] = F::DIMSY::VAL(k);
	else
		for (int k=0; k<F::NVARSI; k++)
			out[k] = F::DIMSX::VAL(k);
	return 0;
}

extern "C" int GetDimsP(int *out) {
	for (int k=0; k<F::NVARSP; k++)
		out[k] = F::DIMSP::VAL(k);
	return 0;
}



