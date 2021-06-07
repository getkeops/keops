

# This is the main entry point for all binders. It takes as inputs :
#   - use_jit : boolean (True or False), use JIT compiling mode or command line
#   - map_reduce_id : string naming the type of map-reduce scheme to be used : either "CpuReduc", "GpuReduc1D_FromDevice", ...
#   - red_formula_string : string expressing the formula, such as "Sum_Reduction((Exp(Minus(Sum(Square((Var(0,3,0) / Var(1,3,1)))))) * Var(2,1,1)),0)",
#   - aliases : list of strings expressing the aliases list, which may be empty,
#   - nargs : integer specifying the number of arguments for the call to the routine, 
#   - dtype : string specifying the float type of the arguments (currently "float" or "double")
#   - dtypeacc : string specifying the float type of the accumulator of the reduction (currently "float" or "double")
#   - sum_scheme_string : string specifying the type of accumulation for summation reductions : either "direct_sum", "block_sum" or "kahan_scheme".

# It returns :
#   * if use_jit is True:
#       - dllname : string, file name of the dll to be called for performing the reduction,
#       - low_level_code_file : string, file name of the low level code file to be passed to the dll,
#       - tagI : integer, 0 or 1, specifying if reduction muust be performed over i or j indices,
#       - dim : integer, dimension of the output tensor.
#       - dimy : integer, total dimension of the j indexed variables.
#   * if use_jit is False:
#       - dllname : string, file name of the dll to be called for performing the reduction,
#       - tagI : integer, 0 or 1, specifying if reduction muust be performed over i or j indices,
#       - dim : integer, dimension of the output tensor.

# It can be used as a Python function or as a standalone Python script (in which case it prints the outputs):
#   - example (as Python function) : 
#       get_keops_dll(False, "CpuReduc", "Sum_Reduction((Exp(Minus(Sum(Square((Var(0,3,0) / Var(1,3,1)))))) * Var(2,1,1)),0)", [], 3, "float", "float", "block_sum")
#   - example (as Python script) : 
#       python get_keops_dll.py False CpuReduc "Sum_Reduction((Exp(Minus(Sum(Square((Var(0,3,0) / Var(1,3,1)))))) * Var(2,1,1)),0)" "[]" 3 float float block_sum

import sys
from keops.python_engine.formulas.variables.Zero import Zero
from keops.python_engine.formulas.reductions import *
from keops.python_engine.mapreduce import *


def get_keops_dll(use_jit, map_reduce_id, *args):
    map_reduce_class = eval(map_reduce_id)
    map_reduce_obj = map_reduce_class(use_jit, *args)
    
    # detecting the case of formula being equal to zero, to bypass reduction.
    rf = map_reduce_obj.red_formula
    if isinstance(rf, Zero_Reduction) or (isinstance(rf.formula, Zero) and isinstance(rf, Sum_Reduction)):
        map_reduce_obj = map_reduce_class.AssignZero(*args)
        
    res = map_reduce_obj.get_dll_and_params()
    
    if use_jit:
        return res["dllname"], res["low_level_code_file"], res["tagI"], res["dim"], res["dimy"]
    else:
        return res["dllname"], res["tagI"], res["dim"]


if __name__ == "__main__":
    argv = sys.argv[1:]

    argdict = {
            'use_jit' : bool,
            'map_reduce_id' : str,
            'red_formula_string' : str, 
            'aliases' : list,
            'nargs' : int, 
            'dtype' : str, 
            'dtypeacc' : str, 
            'sum_scheme_string' : str
            }
    
    if len(argv) != len(argdict):
        raise ValueError(f'Invalid call to Python script {sys.argv[0]}. There should be {len(argdict)} arguments corresponding to:\n{list(argdict.keys())}')
    
    for k, key in enumerate(argdict):
        argtype = argdict[key]
        argval = argv[k] if argtype == str else eval(argv[k])
        if not isinstance(argval,argtype):
            raise ValueError(f'Invalid call to Python script {sys.argv[0]}. Argument number {k+1} ({key}) should be of type {argtype} but is of type {type(argval)}')
        argdict[key] = argval
        
    res = get_keops_dll(argdict["use_jit"], argdict["map_reduce_id"], *list(argdict.values())[2:])
    for item in res:
        print(item)
    
