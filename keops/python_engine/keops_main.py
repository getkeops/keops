
# usage : python keops_main.py <map_reduce_id> <red_formula_string> <aliases> <nargs> <dtype> <dtypeacc> <sum_scheme_string>

# example : 
#    python keops_main.py CpuReduc "Sum_Reduction((Exp(Minus(Sum(Square((Var(0,3,0) / Var(1,3,1)))))) * Var(2,1,1)),0)" [] 3 float float block_sum

import sys
from map_reduce import *

argv = sys.argv[1:]

argdict = {
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
        
map_reduce_class = eval(argdict["map_reduce_id"])
map_reduce_obj = map_reduce_class(*list(argdict.values())[1:])

res = map_reduce_obj.get_dll_and_params()

print(res)