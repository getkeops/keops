#pragma once

extern "C" {
int GetFormulaConstants(int*);
int GetIndsI(int*);
int GetIndsJ(int*);
int GetIndsP(int*);
int GetDimsX(int*);
int GetDimsY(int*);
int GetDimsP(int*);
};

namespace keops_binders {

/////////////////////////////////////////////////////////////////////////////////
//                           Keops                                             //
/////////////////////////////////////////////////////////////////////////////////

int formula_constants[9]; 
int dummy1 = GetFormulaConstants(formula_constants);
int nminargs = formula_constants[0];
int tagIJ = formula_constants[1];
int pos_first_argI = formula_constants[2];
int pos_first_argJ= formula_constants[3];
int nvars = formula_constants[4];
int nvarsI = formula_constants[5];
int nvarsJ = formula_constants[6];
int nvarsP = formula_constants[7];
int dimout = formula_constants[8];

std::vector<int> indsI(nvarsI), indsJ(nvarsJ), indsP(nvarsP);
int dummy2 = GetIndsI(indsI.data());
int dummy3 = GetIndsJ(indsJ.data());
int dummy4 = GetIndsP(indsP.data());
std::vector<int> dimsX(nvarsI), dimsY(nvarsJ), dimsP(nvarsP);
int dummy5 = GetDimsX(dimsX.data());
int dummy6 = GetDimsY(dimsY.data());
int dummy7 = GetDimsP(dimsP.data());



//const std::string f = PrintReduction< F >();

}
