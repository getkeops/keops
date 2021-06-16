from keops.python_engine.utils.code_gen_utils import Var_loader, c_variable, new_c_varname, pointer
from keops.python_engine.reductions import *


class MapReduce:
    # base class for map-reduce schemes
    
    def __init__(self, red_formula_string, aliases, nargs, dtype, dtypeacc, sum_scheme_string, tagHostDevice, tagCpuGpu, tag1D2D):
        self.red_formula_string = red_formula_string
        self.aliases = aliases
                
        self.red_formula = getReduction(red_formula_string, aliases)
        
        self.dtype = dtype
        self.dtypeacc = dtypeacc
        self.nargs = nargs
        self.sum_scheme_string = sum_scheme_string
        self.tagHostDevice, self.tagCpuGpu, self.tag1D2D = tagHostDevice, tagCpuGpu, tag1D2D
    
    def get_code(self):       
        
        self.headers = "#define C_CONTIGUOUS 1\n"     
        
        red_formula = self.red_formula
        formula = red_formula.formula
        dtype = self.dtype
        dtypeacc = self.dtypeacc
        nargs = self.nargs
        self.sum_scheme = eval(self.sum_scheme_string)(red_formula, dtype)

        self.varloader = Var_loader(red_formula)
        
        self.i = i = c_variable("int", "i")
        self.j = j = c_variable("int", "j")

        nx = c_variable("int", "nx")
        ny = c_variable("int", "ny")
        
        self.xi = c_array(dtype, self.varloader.dimx, "xi")
        self.param_loc = c_array(dtype, self.varloader.dimp, "param_loc")
        
        argnames = new_c_varname("arg", nargs)
        self.args = c_variable(pointer(dtype), argnames)
        
        argshapenames = new_c_varname("argshape", nargs)
        self.argshapes = c_variable(pointer("int"), argshapenames)
        
        self.acc = c_array(dtypeacc, red_formula.dimred, "acc")
        self.fout = c_array(dtype, formula.dim, "fout")
        self.outi = c_array(dtype, red_formula.dim, f"(out + i * {red_formula.dim})")
















