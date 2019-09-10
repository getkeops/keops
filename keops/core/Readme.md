This directory contains the core files of KeOps.

 - autodiff   : the headers with needed classes to differentiate automatically a formula
 - formulas   : the files with the math operators that can be used in a formula
 - mapreduce  : the cpu and gpu (cuda) kernels used to make the computation in KeOps
 - pack       : the internal template library used to generate KeOps formulas
 - reductions : the different types of reduction that can be used in KeOps formulas
 - utils      : some useful functions

 - link_autodiff files contains the extern "C" routine exposed in the shared libraries