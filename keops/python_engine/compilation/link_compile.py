import os, time
from ctypes import create_string_buffer, CDLL

from keops.python_engine.utils.code_gen_utils import get_hash_name
from keops.python_engine.config import build_path

class link_compile:
    
    # base class for compiling the map_reduce schemes and
    # providing the dll to KeOps bindings.
    
    def __init__(self, use_jit):
        self.use_jit = use_jit
        self.gencode_filename = get_hash_name(type(self), self.red_formula_string, self.aliases, self.nargs, self.dtype, self.dtypeacc, self.sum_scheme_string)
        self.gencode_file = build_path + os.path.sep + self.gencode_filename + "." + self.source_code_extension
        self.info_file = self.gencode_file + ".nfo"      
        
        if use_jit:
            # these are used for JIT compiling mode
            self.low_level_code_file = (build_path + os.path.sep + self.gencode_filename + "." + self.low_level_code_extension).encode('utf-8')
            self.my_c_dll = CDLL(self.jit_binary)
        else:
            # these are used for command line compiling mode
            self.dllname = self.gencode_file + ".so"  
            self.compile_command = f"{self.compiler} {' '.join(self.compile_options)} {self.gencode_file} -o {self.dllname}"
    
    def save_info(self):
        f = open(self.info_file,"w")
        f.write(f"dim={self.dim}\ntagI={self.tagI}\ndimy={self.dimy}")
        f.close()
    
    def read_info(self):
        f = open(self.info_file,"r")
        string = f.read()
        f.close()
        tmp = string.split("\n")
        if len(tmp)!= 3:
            raise ValueError("incorrect info file")
        tmp_dim, tmp_tag, tmp_dimy = tmp[0].split("="), tmp[1].split("="), tmp[2].split("=")
        if len(tmp_dim)!=2 or tmp_dim[0]!="dim" or len(tmp_tag)!=2 or tmp_tag[0]!="tagI" or len(tmp_dimy)!=2 or tmp_dimy[0]!="dimy":
            raise ValueError("incorrect info file")
        self.dim = eval(tmp_dim[1])
        self.tagI = eval(tmp_tag[1])
        self.dimy = eval(tmp_dimy[1])
        
    def write_code(self):
        f = open(self.gencode_file,"w")
        f.write(self.code)
        f.close()
        
    def compile_code(self):        
        self.tagI = self.red_formula.tagI
        self.dim = self.red_formula.dim
        if self.use_jit:
            self.get_code(for_jit=True)
            self.my_c_dll.Compile(create_string_buffer(self.low_level_code_file), create_string_buffer(self.code.encode('utf-8')))
        else:
            self.get_code(for_jit=False)
            self.write_code()
            os.system(self.compile_command)
        self.dimy = self.varloader.dimy
            
    def get_dll_and_params(self):
        file_to_check = self.low_level_code_file if self.use_jit else self.dllname
        if not os.path.exists(file_to_check):
            print("[KeOps] Compiling formula :", self.red_formula, "...", flush=True, end="")
            start = time.time()
            self.compile_code()
            self.save_info()
            elapsed = time.time()-start
            print("Done ({:.2f} s)".format(elapsed))
        else:
            self.read_info()
        if self.use_jit:
            return dict(dllname=self.jit_binary, low_level_code_file=self.low_level_code_file, tagI=self.tagI, dim=self.dim, dimy=self.dimy)
        else:
            return dict(dllname=self.dllname, tagI=self.tagI, dim=self.dim)    
        
        
