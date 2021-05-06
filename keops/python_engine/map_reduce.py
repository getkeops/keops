from code_gen_utils import *
from sum_schemes import *
from reductions import *
from link_compile import *


class map_reduce:
    # base class for map-reduce schemes
    
    def __init__(self, red_formula_string, aliases, nargs, dtype, dtypeacc, sum_scheme_string):
        self.red_formula_string = red_formula_string
        self.aliases = aliases
                
        self.red_formula = getReduction(red_formula_string, aliases)
        
        self.dtype = dtype
        self.dtypeacc = dtypeacc
        self.nargs = nargs
        self.sum_scheme_string = sum_scheme_string
    
    def get_code(self):       
        
        self.headers = "#define C_CONTIGUOUS 1\n"     
        
        red_formula = self.red_formula
        formula = red_formula.formula
        dtype = self.dtype
        dtypeacc = self.dtypeacc
        nargs = self.nargs
        self.sum_scheme = eval(self.sum_scheme_string)(red_formula, dtype)

        self.varloader = varloader = Var_loader(red_formula)
        
        self.i = i = c_variable("int", "i")
        self.j = j = c_variable("int", "j")

        nx = c_variable("int", "nx")
        ny = c_variable("int", "ny")
        
        self.xi = xi = c_array(dtype, self.varloader.dimx, "xi")
        self.param_loc = param_loc = c_array(dtype, self.varloader.dimp, "param_loc")
        argnames = new_c_varname("arg", nargs)
        self.args = args = c_variable(pointer(dtype), argnames)
        self.acc = acc = c_array(dtypeacc, red_formula.dimred, "acc")
        self.fout = fout = c_array(dtype, formula.dim, "fout")
        self.outi = c_array(dtype, red_formula.dim, f"(out + i * {red_formula.dim})") 



class CpuAssignZero(map_reduce, Cpu_link_compile):
    # class for generating the final C++ code, Cpu version
    
    def __init__(self, *args):
        map_reduce.__init__(self, *args)
        Cpu_link_compile.__init__(self)
    
    def get_code(self):
        
        super().get_code()
        
        outi = self.outi
        dtype = self.dtype
        args = self.args
        
        self.headers += c_include("omp.h")

        self.code = f"""
                        {self.headers}

                        extern "C" int AssignZeroCpu(int nx, int ny, {dtype}* out, {signature_list(args)}) {{
                            #pragma omp parallel for
                            for (int i = 0; i < nx; i++) {{
                                {outi.assign(c_zero_float)}
                            }}
                            return 0;
                        }}
                        
                        extern "C" int launch_keops(int nx, int ny, int device_id, int *ranges, {dtype}* out, {signature_list(args)}) {{
                            return AssignZeroCpu(nx, ny, out, {call_list(args)});
                        }}
                    """


class GpuAssignZero(map_reduce, Gpu_link_compile):
    # class for generating the final C++ code, Gpu version
    
    def __init__(self, *args):
        map_reduce.__init__(self, *args)
        Gpu_link_compile.__init__(self)
    
    def get_code(self):
        
        super().get_code()
        
        outi = self.outi
        dtype = self.dtype
        args = self.args
        varloader = self.varloader
        
        if dtype == "half2":
            self.headers += c_include("cuda_fp16.h")
            
        self.code = f"""
                        {self.headers}

                        __global__ void AssignZeroGpu(int nx, int ny, {dtype} *out, {signature_list(args)}) {{
    
                          // get the index of the current thread
                          int i = blockIdx.x * blockDim.x + threadIdx.x;

                          if (i < nx) {{
                            {outi.assign(c_zero_float)}
                          }}

                        }}



                        extern "C" __host__ int launch_keops(int nx, int ny, int device_id, int *ranges, {dtype} *out, {signature_list(args)}) {{

                            // device_id is provided, so we set the GPU device accordingly
                            // Warning : is has to be consistent with location of data
                            cudaSetDevice(device_id);
	
                            // Compute on device : grid and block are both 1d

                            //SetGpuProps(device_id);

                            dim3 blockSize;

                            blockSize.x = 32;
	
                            dim3 gridSize;
                            gridSize.x = nx / blockSize.x + (nx % blockSize.x == 0 ? 0 : 1);

                            AssignZeroGpu <<< gridSize, blockSize, blockSize.x * {varloader.dimy} * sizeof({dtype}) >>> (nx, ny, out, {call_list(args)});
    
                            // block until the device has completed
                            cudaDeviceSynchronize();

                            //CudaCheckError();

                            return 0;
                        }}
                    """



            
            
class CpuReduc(map_reduce, Cpu_link_compile):
    # class for generating the final C++ code, Cpu version
    
    AssignZero = CpuAssignZero

    def __init__(self, *args):
        map_reduce.__init__(self, *args)
        Cpu_link_compile.__init__(self)
        
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
        
        self.headers += c_include("cmath", "omp.h")

        self.code = f"""
                        {self.headers}
                        int CpuConv(int nx, int ny, {dtype}* out, {signature_list(args)}) {{
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
                                    {sum_scheme.periodic_accumulate_temporary(acc, j)}
                                }}
                                {sum_scheme.final_operation(acc)}
                                {red_formula.FinalizeOutput(acc, outi, i)}
                            }}
                            return 0;
                        }}
                        
                        extern "C" int launch_keops(int nx, int ny, int device_id, int *ranges, {dtype}* out, {signature_list(args)}) {{
                            return CpuConv(nx, ny, out, {call_list(args)});
                        }}
                    """


class GpuReduc1D_FromDevice(map_reduce, Gpu_link_compile):
    # class for generating the final C++ code, Gpu version
    
    AssignZero = GpuAssignZero

    def __init__(self, *args):
        map_reduce.__init__(self, *args)
        Gpu_link_compile.__init__(self)
        
    def get_code(self):
        
        super().get_code()
        
        red_formula = self.red_formula
        dtype = self.dtype
        varloader = self.varloader
        
        i = c_variable("int", "i")
        j = c_variable("int", "j")
        fout = self.fout
        outi = self.outi
        acc = self.acc
        args = self.args
        sum_scheme = self.sum_scheme
        
        param_loc = self.param_loc
        xi = self.xi
        yjloc = c_array(dtype, varloader.dimy, f"(yj + threadIdx.x * {varloader.dimy})")
        yjrel = c_array(dtype, varloader.dimy, "yjrel")
        table = varloader.table(self.xi, yjrel, self.param_loc)
        jreltile = c_variable("int", "(jrel + tile * blockDim.x)")
        
        if dtype == "half2":
            self.headers += c_include("cuda_fp16.h")
            
        self.code = f"""
                        {self.headers}

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



                        extern "C" __host__ int launch_keops(int nx, int ny, int device_id, int *ranges, {dtype} *out, {signature_list(args)}) {{

                            // device_id is provided, so we set the GPU device accordingly
                            // Warning : is has to be consistent with location of data
                            cudaSetDevice(device_id);
	
                            // Compute on device : grid and block are both 1d

                            //SetGpuProps(device_id);

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
















