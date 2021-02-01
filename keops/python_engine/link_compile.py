from utils import *
import os
from ctypes import c_int, c_float, c_double, c_void_p, CDLL, POINTER
import time

base_dir_path = os.path.dirname(os.path.realpath(__file__)) + os.path.sep
template_path = base_dir_path + "templates"
build_path = base_dir_path + "build" + os.path.sep
    
class genred:
    # base class for compiling and launching reductions
    
    library = {}
    
    def __init__(self, red_formula, dtype, dtypeacc, nargs, sum_scheme):
        # - red_formula is instance of Reduction class
        # - dtype and dtypeacc are strings 
        self.gencode_filename = get_hash_name(type(self), red_formula, dtype, dtypeacc, nargs, sum_scheme)
        self.dllname = build_path + os.path.sep + self.gencode_filename + "_" + self.source_code_extension + ".so"
        self.red_formula = red_formula
        self.dtype = dtype
        self.dtypeacc = dtypeacc
        self.nargs = nargs
        self.sum_scheme = sum_scheme
        if self.gencode_filename in genred.library:
            rec = genred.library[self.gencode_filename]  
            self.dll = rec["dll"]
            self.tagI = rec["tagI"]
            self.dim = rec["dim"]
        else:
            self.load_dll()
    
    def get_code(self):  
        self.template_source_file = template_path + os.path.sep + self.template_source_filename
        self.gencode_file = build_path + os.path.sep + self.gencode_filename + "." + self.source_code_extension
        self.compile_command = f"{self.compiler} {' '.join(self.compile_options)} {self.gencode_file} -o {self.dllname}"
        sum_schemes_dict = {"direct_sum" : 0, "block_sum" : 1, "kahan_scheme" : 2}
        self.compile_options += [f"-DSUM_SCHEME={sum_schemes_dict[self.sum_scheme]}"]
          
        red_formula = self.red_formula
        formula = red_formula.formula
        dtype = self.dtype
        dtypeacc = self.dtypeacc
        nargs = self.nargs

        self.varloader = varloader = Var_loader(red_formula)
        
        self.i = i = c_variable("i", "int")
        self.j = j = c_variable("j", "int")

        nx = c_variable("nx", "int")
        ny = c_variable("ny", "int")
        
        self.xi = xi = c_array("xi", dtype, self.varloader.dimx)
        self.param_loc = param_loc = c_array("param_loc", dtype, self.varloader.dimp)
        argnames = new_c_varname("arg", nargs)
        self.args = args = c_variable(argnames, pointer(dtype))
        self.acc = acc = c_array("acc", dtypeacc, red_formula.dimred)
        self.tmp = tmp = c_array("tmp", dtype, red_formula.dimred)
        self.tmp_kahan = tmp_kahan = c_array("tmp", dtype, red_formula.dim_kahan)
        self.fout = fout = c_array("fout", dtype, formula.dim)
        self.outi = c_array(f"(out + i * {red_formula.dim})", dtype, red_formula.dim) 
        
    def get_template_code(self):
        # reads the source_file associated with the class
        f = open(self.template_source_file)
        template_code = f.read()
        f.close()
        return template_code 
        
    def write_code(self):
        f = open(self.gencode_file,"w")
        f.write(self.code)
        f.close()
    
    def compile_code(self):
        self.get_code()
        self.write_code()
        os.system(self.compile_command)
    
    def load_dll(self):
        if not os.path.exists(self.dllname):
            print("compiling dll...", end="", flush=True)
            start = time.time()
            self.compile_code()
            elapsed = time.time()-start
            print("done ({:.2f} s)".format(elapsed))
        self.dll = CDLL(self.dllname)
        genred.library[self.gencode_filename] = { "dll":self.dll, "tagI":self.red_formula.tagI, "dim":self.red_formula.dim }
        ctype = eval(f"c_{self.dtype}")
        self.dll.argtypes = [c_int, c_int, POINTER(ctype)] + [POINTER(ctype)]*self.nargs
        
    def __call__(self, nx, ny, out, *args):
        if self.dll is None:
            self.load_dll()
        c_args = [c_void_p(x.data_ptr()) for x in args]
        self.dll.Eval(c_int(nx), c_int(ny), c_void_p(out.data_ptr()), *c_args)
            
            
class CpuReduc(genred):
    # class for generating the final C++ code, Cpu version
    
    template_source_filename = "CpuReduc.cpp" 
    source_code_extension = "cpp"
    compiler = "g++"
    compile_options = ["-shared", "-O3"]
    
    def get_code(self):

        super().get_code()
        
        i = self.i
        j = self.j
        fout = self.fout
        outi = self.outi
        acc = self.acc
        tmp = self.tmp
        tmp_kahan = self.tmp_kahan
        args = self.args
        table = self.varloader.direct_table(args, i, j)
        
        self.code = eval("f'''"+self.get_template_code()+"'''")

     

class GpuReduc1D(genred):
    # class for generating the final C++ code, Gpu version
    
    template_source_filename = "GpuReduc1D.cu" 
    source_code_extension = "cu"
    compiler = "nvcc"
    compile_options = ["-shared", "-Xcompiler", "-fPIC", "-O3"]
    
    def get_code(self):
        
        super().get_code()
        
        red_formula = self.red_formula
        dtype = self.dtype
        varloader = self.varloader
        
        i = c_variable("i", "int")
        j = c_variable("j", "int")
        fout = self.fout
        outi = self.outi
        acc = self.acc
        tmp = self.tmp
        tmp_kahan = self.tmp_kahan
        args = self.args
        
        param_loc = self.param_loc
        xi = self.xi
        yjloc = c_array(f"(yj + threadIdx.x * {varloader.dimy})", dtype, varloader.dimy)
        yjrel = c_array("yjrel", dtype, varloader.dimy)
        table = varloader.table(self.xi, yjrel, self.param_loc)
        jreltile = c_variable("(jrel + tile * blockDim.x)","int")
        ind_pack_half2 = c_variable("__floats2half2_rn(2*ind,2*ind+1)","half2")
        
        self.code = eval("f'''"+self.get_template_code()+"'''")


