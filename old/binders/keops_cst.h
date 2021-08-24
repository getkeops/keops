#pragma once

extern "C" {
int GetFormulaString(std::string&);
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

int formula_constants[8]; 
int dummy1 = GetFormulaConstants(formula_constants);
int keops_nminargs = formula_constants[0];
int keops_tagIJ = formula_constants[1];
int keops_pos_first_argI = formula_constants[2];
int keops_pos_first_argJ = formula_constants[3];
int keops_nvarsI = formula_constants[4];
int keops_nvarsJ = formula_constants[5];
int keops_nvarsP = formula_constants[6];
int keops_dimout = formula_constants[7];


std::vector<int> keops_indsI(keops_nvarsI), keops_indsJ(keops_nvarsJ), keops_indsP(keops_nvarsP);
int dummy2 = GetIndsI(keops_indsI.data());
int dummy3 = GetIndsJ(keops_indsJ.data());
int dummy4 = GetIndsP(keops_indsP.data());
std::vector<int> keops_dimsX(keops_nvarsI), keops_dimsY(keops_nvarsJ), keops_dimsP(keops_nvarsP);
int dummy5 = GetDimsX(keops_dimsX.data());
int dummy6 = GetDimsY(keops_dimsY.data());
int dummy7 = GetDimsP(keops_dimsP.data());


std::string keops_formula_string;
int dummy8 = GetFormulaString(keops_formula_string);


}
