import os, glob, time
from ctypes import c_int, c_float, c_double, c_void_p, CDLL, POINTER
from utils import *
from operations import Zero

base_dir_path = os.path.dirname(os.path.realpath(__file__)) + os.path.sep
template_path = base_dir_path + "templates"
build_path = base_dir_path + "build" + os.path.sep
os.makedirs(build_path, exist_ok=True)



    
        
        
class link_compile:
    # base class for compiling reductions
    
    def __init__(self, map_reduce_obj, red_formula_string, nargs, dtype, *params):
        self.gencode_filename = get_hash_name(type(map_reduce_obj), red_formula_string, nargs, dtype, *params)
        self.gencode_file = build_path + os.path.sep + self.gencode_filename + "." + self.source_code_extension
        print(type(map_reduce_obj))
        print(type(map_reduce_obj.red_formula))
        self.dllname = self.gencode_file + ".so" if not isinstance(map_reduce_obj.red_formula.formula,Zero) else "zero"
        self.info_file = self.gencode_file + ".nfo"
        self.map_reduce_obj = map_reduce_obj
        self.red_formula_string = red_formula_string
        self.nargs = nargs
        self.dtype = dtype
        self.params = params
        self.compile_command = f"{self.compiler} {' '.join(self.compile_options)} {self.gencode_file} -o {self.dllname}"
        
    def write_code(self, map_reduce_obj):
        f = open(self.gencode_file,"w")
        f.write(map_reduce_obj.code)
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
        self.tagI = self.map_reduce_obj.red_formula.tagI
        self.dim = self.map_reduce_obj.red_formula.dim
        self.map_reduce_obj.get_code()
        self.write_code(self.map_reduce_obj)
        os.system(self.compile_command)
        
    def get_dll_and_params(self):
        if self.dllname != "zero" and not os.path.exists(self.dllname):
            print("compiling dll...", end="", flush=True)
            start = time.time()
            self.compile_code()
            self.save_info()
            elapsed = time.time()-start
            print("done ({:.2f} s)".format(elapsed))
        else:
            self.read_info()
        return dict(dllname=self.dllname, tagI=self.tagI, dim=self.dim)
    

            
            
class Cpu_link_compile(link_compile):

    source_code_extension = "cpp"
    compiler = "g++"
    compile_options = ["-shared", "-O3"]
    
    def __call__(self, nx, ny, out, *args):
        if self.dll is None:
            self.load_dll()
        c_args = [c_void_p(x.data_ptr()) for x in args]
        self.dll.Eval(c_int(nx), c_int(ny), c_void_p(out.data_ptr()), *c_args)


class Gpu_link_compile(link_compile):
    
    source_code_extension = "cu"
    compiler = "nvcc"
    compile_options = ["-shared", "-Xcompiler", "-fPIC", "-O3"]

    def __call__(self, nx, ny, out, *args):
        if self.dll is None:
            self.load_dll()
        c_args = [c_void_p(x.data_ptr()) for x in args]
        device_id = out.device.index
        self.dll.Eval(c_int(nx), c_int(ny), c_int(device_id), c_void_p(out.data_ptr()), *c_args)

