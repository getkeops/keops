import os, time

from keops.python_engine.config import build_path
from keops.python_engine.utils.code_gen_utils import get_hash_name


# flag for OpenMP support
use_OpenMP = True
    
        
        
class link_compile:
    
    # base class for compiling the map_reduce schemes and
    # providing the dll to KeOps bindings.
    
    def __init__(self):
        self.gencode_filename = get_hash_name(type(self), self.red_formula_string, self.aliases, self.nargs, self.dtype, self.dtypeacc, self.sum_scheme_string)
        self.gencode_file = build_path + os.path.sep + self.gencode_filename + "." + self.source_code_extension
        self.dllname = self.gencode_file + ".so"
        self.info_file = self.gencode_file + ".nfo"
        self.compile_command = f"{self.compiler} {' '.join(self.compile_options)} {self.gencode_file} -o {self.dllname}"
        
    def write_code(self):
        f = open(self.gencode_file,"w")
        f.write(self.code)
        f.close()
    
    def save_info(self):
        f = open(self.info_file,"w")
        f.write(f"dim={self.dim}\ntagI={self.tagI}")
        f.close()
    
    def read_info(self):
        f = open(self.info_file,"r")
        string = f.read()
        f.close()
        tmp = string.split("\n")
        if len(tmp)!= 2:
            raise ValueError("incorrect info file")
        tmp_dim, tmp_tag = tmp[0].split("="), tmp[1].split("=")
        if len(tmp_dim)!=2 or tmp_dim[0]!="dim" or len(tmp_tag)!=2 or tmp_tag[0]!="tagI":
            raise ValueError("incorrect info file")
        self.dim = eval(tmp_dim[1])
        self.tagI = eval(tmp_tag[1])
        
    def compile_code(self):        
        self.tagI = self.red_formula.tagI
        self.dim = self.red_formula.dim
        self.get_code()
        self.write_code()
        os.system(self.compile_command)
        
    def get_dll_and_params(self):
        if not os.path.exists(self.dllname):
            print("[KeOps] Compiling formula :", self.red_formula, "...", flush=True)
            start = time.time()
            self.compile_code()
            self.save_info()
            elapsed = time.time()-start
            print("Done ({:.2f} s)".format(elapsed))
        else:
            self.read_info()
        return dict(dllname=self.dllname, tagI=self.tagI, dim=self.dim)
    

            
            
class Cpu_link_compile(link_compile):

    source_code_extension = "cpp"
    
    compiler = "g++"
    compile_options = ["-shared", "-fPIC", "-O3", "-flto"]
    
    if use_OpenMP:
        import platform
        if platform.system()=="Darwin":
            compile_options += ["-Xclang -fopenmp", "-lomp"]
            # warning : this is unsafe hack for OpenMP support on mac...
            os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        else:
            compile_options += ["-fopenmp", "-fno-fat-lto-objects"]
    


class Gpu_link_compile(link_compile):
    
    source_code_extension = "cu"
    compiler = "nvcc"
    compile_options = ["-shared", "-Xcompiler -fPIC", "-O3"]

