from utils import *
import os
from ctypes import c_int, c_float, c_double, c_void_p, CDLL, POINTER

class genred:
    # base class for compiling and launching reductions
    
    def get_code(self):
        # reads the source_file associated with the class and  
        # format the code according to rules given in dict_format
        f = open(self.source_file)
        string = f.read()
        f.close()
        return string.format(**self.dict_format)    
        
    def write_code(self):
        f = open(self.gencode_file,"w")
        f.write(self.get_code())
        f.close()
    
    def compile_code(self):
        self.write_code()
        os.system(self.compile_command)
    
    def load_dll(self):
        self.dll = CDLL(self.dllname)
        ctype = eval(f"c_{self.dtype}")
        self.dll.argtypes = [c_int, c_int, POINTER(ctype)] + [POINTER(ctype)]*self.nargs
        
    def __call__(self, nx, ny, out, *args):
        if self.dll is None:
            self.load_dll()
        c_args = [c_void_p(x.data_ptr()) for x in args]
        self.dll.Eval(c_int(nx), c_int(ny), c_void_p(out.data_ptr()), *c_args)
            
def signature_args(nargs, dtype):
    string = ""
    for k in range(nargs):
        string += f", {dtype}* arg{k}"
    return string

def load_args(nargs):
    string = ""
    for k in range(nargs):
        string += f"args[{k}] = arg{k};\n"
    return string
    
class CpuReduc(genred):
    # class for generating the final C++ code, Cpu version
    
    gencode_file = "test.cpp"
    compile_command = "g++ -dynamiclib -O3 test.cpp -o test.dylib"
    dllname = "test.dylib"
    
    def __init__(self, red_formula, dtype, dtypeacc, nargs):
        # - red_formula is instance of Reduction class
        # - dtype and dtypeacc are strings 
        
        self.source_file = "link_autodiff.cpp" 
        formula = red_formula.formula 
        self.dtype = dtype
        self.dtypeacc = dtypeacc
        self.nargs = nargs
        self.dll = None
        pdtype = pointer(dtype)
        pdtypeacc = pointer(dtypeacc)
        formula = red_formula.formula
        acc = c_variable("acc",pdtypeacc)
        xi = c_variable("xi",pdtype)
        yj = c_variable("yj",pdtype)
        fout = c_variable("fout",pdtype)
        out = c_variable("out",pdtype)
        args = c_variable("args",pointer(pdtype))
        zero = c_variable("0","int")
        i = c_variable("i","int")
        j = c_variable("j","int")
        pp = c_variable("pp",pdtype)
        outi = c_variable(f"(out + i * {red_formula.dim})", out.dtype) 
        inds = GetInds(formula._Vars)
        nminargs = max(inds)+1
        table = [None]*nminargs
        loadp, table = load_vars(red_formula.dimsp, red_formula.indsp, zero, pp, args, table)
        loadx, table = load_vars(red_formula.dimsx, red_formula.indsi, i, xi, args, table)
        loady, table = load_vars(red_formula.dimsy, red_formula.indsj, j, yj, args, table)
        self.dict_format = { 
            "TYPE" : self.dtype,
            "TYPEACC" : self.dtypeacc,
            "args" : signature_args(nargs, self.dtype),
            "loadargs" : load_args(nargs),
            "nargs" : nargs,
            "DIMRED" : red_formula.dimred,
            "DIMP" : red_formula.dimp,
            "DIMX" : red_formula.dimx,
            "DIMY" : red_formula.dimy,
            "DIMOUT" : red_formula.dim,
            "DIMFOUT" : formula.dim,
            "InitializeReduction" : red_formula.InitializeReduction(acc),
            "ReducePairShort" : red_formula.ReducePairShort(acc, fout, j),
            "FinalizeOutput" : red_formula.FinalizeOutput(acc, outi, i),
            "definep" : declare_array(pp,red_formula.dimp),
            "definex" : declare_array(xi,red_formula.dimx),
            "definey" : declare_array(yj,red_formula.dimy),
            "loadp" : loadp,
            "loadx" : loadx,
            "loady" : loady,
            "call" : formula(fout,table),
        }




     

class GpuReduc1D(genred):
    # class for generating the final C++ code, Gpu version
    
    gencode_file = "test.cu"
    compile_command = "nvcc -shared -Xcompiler -fPIC -O3 test.cu -o test.so"
    dllname = "/home/glaunes/Bureau/keops/keops/python_engine/test.so"
    
    def __init__(self, red_formula, dtype, dtypeacc, nargs):
        # - red_formula is instance of Reduction class
        # - dtype and dtypeacc are strings 
        
        self.source_file = "GpuReduc1D.cu" 
        formula = red_formula.formula 
        self.dtype = dtype
        self.dtypeacc = dtypeacc
        self.nargs = nargs
        self.dll = None
        pdtype = pointer(dtype)
        pdtypeacc = pointer(dtypeacc)
        formula = red_formula.formula
        acc = c_variable("acc",pdtypeacc)
        xi = c_variable("xi",pdtype)
        yjloc = c_variable(f"(yj + threadIdx.x * {red_formula.dimy})",pdtype)
        fout = c_variable("fout",pdtype)
        out = c_variable("out",pdtype)
        args = c_variable("args",pointer(pdtype))
        zero = c_variable("0","int")
        i = c_variable("i","int")
        j = c_variable("j","int")
        jreltile = c_variable("(jrel + tile * blockDim.x)","int")
        param_loc = c_variable("param_loc",pdtype)
        outi = c_variable(f"(out + i * {red_formula.dim})", out.dtype) 
        inds = GetInds(formula._Vars)
        nminargs = max(inds)+1
        table = [None]*nminargs
        loadp, table = load_vars(red_formula.dimsp, red_formula.indsp, zero, param_loc, args, table)
        loadx, table = load_vars(red_formula.dimsx, red_formula.indsi, i, xi, args, table)
        loady, table = load_vars(red_formula.dimsy, red_formula.indsj, j, yjloc, args, table)
        self.dict_format = { 
            "TYPE" : self.dtype,
            "TYPEACC" : self.dtypeacc,
            "args" : signature_args(nargs, self.dtype),
            "loadargs" : load_args(nargs),
            "nargs" : nargs,
            "NMINARGS" : nminargs,
            "DIMRED" : red_formula.dimred,
            "DIMP" : red_formula.dimp,
            "DIMX" : red_formula.dimx,
            "DIMY" : red_formula.dimy,
            "DIMOUT" : red_formula.dim,
            "DIMFOUT" : formula.dim,
            "InitializeReduction" : red_formula.InitializeReduction(acc),
            "ReducePairShort" : red_formula.ReducePairShort(acc, fout, jreltile),
            "FinalizeOutput" : red_formula.FinalizeOutput(acc, outi, i),
            "loadp" : loadp,
            "loadx" : loadx,
            "loady" : loady,
            "call" : formula(fout,table),
        }
