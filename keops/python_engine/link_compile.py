from utils import *
from sum_schemes import *
import os
from ctypes import c_int, c_float, c_double, c_void_p, CDLL, POINTER
import time

base_dir_path = os.path.dirname(os.path.realpath(__file__)) + os.path.sep
template_path = base_dir_path + "templates"
build_path = base_dir_path + "build" + os.path.sep
os.makedirs(build_path, exist_ok=True)



    
class genred:
    # base class for compiling and launching reductions
    
    library = {}
    
    def __init__(self, red_formula, dtype, dtypeacc, nargs, sum_scheme_string):
        # - red_formula is instance of Reduction class
        # - dtype and dtypeacc are strings 
        self.gencode_filename = get_hash_name(type(self), red_formula, dtype, dtypeacc, nargs, sum_scheme_string)
        self.dllname = build_path + os.path.sep + self.gencode_filename + "_" + self.source_code_extension + ".so"
        self.red_formula = red_formula
        self.dtype = dtype
        self.dtypeacc = dtypeacc
        self.nargs = nargs
        self.sum_scheme_string = sum_scheme_string
        if self.gencode_filename in genred.library:
            rec = genred.library[self.gencode_filename]  
            self.dll = rec["dll"]
            self.tagI = rec["tagI"]
            self.dim = rec["dim"]
        else:
            self.load_dll()
    
    def get_code(self):  
        self.gencode_file = build_path + os.path.sep + self.gencode_filename + "." + self.source_code_extension
        self.compile_command = f"{self.compiler} {' '.join(self.compile_options)} {self.gencode_file} -o {self.dllname}"
          
        red_formula = self.red_formula
        formula = red_formula.formula
        dtype = self.dtype
        dtypeacc = self.dtypeacc
        nargs = self.nargs
        self.sum_scheme = eval(self.sum_scheme_string)(red_formula, dtype)

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
        self.fout = fout = c_array("fout", dtype, formula.dim)
        self.outi = c_array(f"(out + i * {red_formula.dim})", dtype, red_formula.dim) 
        
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

    source_code_extension = "cpp"
    compiler = "g++"
    compile_options = ["-shared", "-O3"]
    
    def get_code(self):

        super().get_code()
        
        i = self.i
        j = self.j
        dtype = self.dtype
        red_formula = self.red_formula
        fout = self.fout
        outi = self.outi
        acc = self.acc
        args = self.args
        table = self.varloader.direct_table(args, i, j)
        sum_scheme = self.sum_scheme

        self.code = f"""
                        #include <cmath>

                        #ifdef USE_OPENMP
                          #include <omp.h>
                        #endif

                        extern "C" int Eval(int nx, int ny, {dtype}* out, {signature_list(args)}) {{
                            #pragma omp parallel for
                            for (int i = 0; i < nx; i++) {{
                                {fout.declare()}
                                {acc.declare()}
                                {sum_scheme.declare_temporary_accumulator()}
                                {red_formula.InitializeReduction(acc)}
                                {sum_scheme.initialize_temporary_accumulator()}
                                for (int j = 0; j < ny; j++) {{
                                    {red_formula.formula(fout,table)}
                                    {sum_scheme.accumulate_result(acc, fout, j)}
                                    {sum_scheme.periodic_accumulate_temporary(acc)}
                                }}
                                {sum_scheme.final_operation(acc)}
                                {red_formula.FinalizeOutput(acc, outi, i)}
                            }}
                            return 0;
                        }}
                    """

    def __call__(self, nx, ny, out, *args):
        if self.dll is None:
            self.load_dll()
        c_args = [c_void_p(x.data_ptr()) for x in args]
        self.dll.Eval(c_int(nx), c_int(ny), c_void_p(out.data_ptr()), *c_args)


class GpuReduc1D(genred):
    # class for generating the final C++ code, Gpu version

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
        args = self.args
        
        param_loc = self.param_loc
        xi = self.xi
        yjloc = c_array(f"(yj + threadIdx.x * {varloader.dimy})", dtype, varloader.dimy)
        yjrel = c_array("yjrel", dtype, varloader.dimy)
        table = varloader.table(self.xi, yjrel, self.param_loc)
        jreltile = c_variable("(jrel + tile * blockDim.x)","int")
            
        self.code = f"""
                        #define DIRECT_SUM 0
                        #define BLOCK_SUM 1
                        #define KAHAN_SCHEME 2

                        #ifndef USE_HALF
                          #define USE_HALF 0
                        #endif

                        #if USE_HALF
                          #include <cuda_fp16.h>
                        #endif

                        __global__ void GpuConv1DOnDevice(int nx, int ny, {dtype} *out, {signature_list(args)}) {{
    
                          // get the index of the current thread
                          int i = blockIdx.x * blockDim.x + threadIdx.x;

                          // declare shared mem
                          extern __shared__ {dtype} yj[];

                          // load parameters variables from global memory to local thread memory
                          {param_loc.declare()}
                          {varloader.load_vars("p", param_loc, args)}

                          {fout.declare()}
                          {xi.declare()}
                          {acc.declare()}
                          {sum_scheme.declare_temporary_accumulator()}
	  
                          if (i < nx) {{
                            {red_formula.InitializeReduction(acc)} // acc = 0
                            {sum_scheme.initialize_temporary_accumulator_first_init()}
                            {varloader.load_vars('i', xi, args, row_index=i)} // load xi variables from global memory to local thread memory
                          }}

                          for (int jstart = 0, tile = 0; jstart < ny; jstart += blockDim.x, tile++) {{

                            // get the current column
                            int j = tile * blockDim.x + threadIdx.x;

                            if (j < ny) {{ // we load yj from device global memory only if j<ny
                              {varloader.load_vars("j", yjloc, args, row_index=j)} 
                            }}
                            __syncthreads();

                            if (i < nx) {{ // we compute x1i only if needed
                              {dtype} * yjrel = yj;
                              {sum_scheme.initialize_temporary_accumulator_block_init()}
                              for (int jrel = 0; (jrel < blockDim.x) && (jrel < ny - jstart); jrel++, yjrel += {varloader.dimy}) {{
                                {red_formula.formula(fout,table)} // Call the function, which outputs results in fout
                                {sum_scheme.accumulate_result(acc, fout, jreltile)}
                              }}
                              {sum_scheme.final_operation(acc)}
                            }}
                            __syncthreads();
                          }}
                          if (i < nx) {{
                            {red_formula.FinalizeOutput(acc, outi, i)} 
                          }}

                        }}



                        extern "C" __host__ int Eval(int nx, int ny, int device_id, {dtype} *out, {signature_list(args)}) {{

                            // device_id is provided, so we set the GPU device accordingly
                            // Warning : is has to be consistent with location of data
                            cudaSetDevice(device_id);
	
                            // Compute on device : grid and block are both 1d

                            //SetGpuProps(devise_id);

                            dim3 blockSize;

                            blockSize.x = 32;
	
                            dim3 gridSize;
                            gridSize.x = nx / blockSize.x + (nx % blockSize.x == 0 ? 0 : 1);

                            GpuConv1DOnDevice <<< gridSize, blockSize, blockSize.x * {varloader.dimy} * sizeof({dtype}) >>> (nx, ny, out, {call_list(args)});
    
                            // block until the device has completed
                            cudaDeviceSynchronize();

                            //CudaCheckError();

                            return 0;
                        }}
                    """

    def __call__(self, nx, ny, out, *args):
        if self.dll is None:
            self.load_dll()
        c_args = [c_void_p(x.data_ptr()) for x in args]
        device_id = out.device.index
        self.dll.Eval(c_int(nx), c_int(ny), c_int(device_id), c_void_p(out.data_ptr()), *c_args)

