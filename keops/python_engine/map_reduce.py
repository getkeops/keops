from utils import *
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


class CpuReduc_ranges(map_reduce, Cpu_link_compile):
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
        
        tagI, tagJ = self.red_formula.tagI, self.red_formula.tagJ
        size_i, size_j, size_p = (len(self.red_formula.Vars(cat)) for cat in tagI, tagJ, 2)
        
        self.headers += c_include("cmath", "omp.h")

        self.code = f"""
                        {self.headers}
                        static int CpuConv(int nx, int ny, {dtype}* out, {signature_list(args)}) {{
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

        self.code = f"""
                    {self.headers}
                    #define __INDEX__ int32_t
                    
                    {define_fill_shapes_function(red_formula)}
                    
                    static int CpuConv_ranges(int nx, int ny, int nbatchdims, int* shapes,
                                              int nranges_x, int nranges_y, __INDEX__** ranges, 
                                              {dtype}* out, {signature_list(args)}) {{
                        
                        // Separate and store the shapes of the "i" and "j" variables + parameters --------------
                        //
                        // shapes is an array of size (1+nargs)*(nbatchdims+3), which looks like:
                        // [ A, .., B, M, N, D_out]  -> output
                        // [ A, .., B, M, 1, D_1  ]  -> "i" variable
                        // [ A, .., B, 1, N, D_2  ]  -> "j" variable
                        // [ A, .., B, 1, 1, D_3  ]  -> "parameter"
                        // [ A, .., 1, M, 1, D_4  ]  -> N.B.: we support broadcasting on the batch dimensions!
                        // [ 1, .., 1, M, 1, D_5  ]  ->      (we'll just ask users to fill in the shapes with *explicit* ones)
                        
                        {shapes_i.declare()}
                        int shapes_i[({size_i}) * (nbatchdims + 1)], shapes_j[{size_j} * (nbatchdims + 1)],
                        shapes_p[{size_p} * (nbatchdims + 1)];
                        
                        // First, we fill shapes_i with the "relevant" shapes of the "i" variables,
                        // making it look like, say:
                        // [ A, .., B, M]
                        // [ A, .., 1, M]
                        // [ A, .., A, M]
                        // Then, we do the same for shapes_j, but with "N" instead of "M".
                        // And finally for the parameters, with "1" instead of "M".
                        fill_shapes(nbatchdims, shapes, shapes_i, shapes_j, shapes_p);
                        
                        // Actual for-for loop -----------------------------------------------------
                        
                        TYPE pp[DIMP];
                        load< DIMSP, INDSP >(0, pp, args);  // If nbatchdims == 0, the parameters are fixed once and for all
                        
                        // Set the output to zero, as the ranges may not cover the full output -----
                        __TYPEACC__ acctmp[DIMRED];
                        for (int i = 0; i < nx; i++) {
                        typename FUN::template InitializeReduction< __TYPEACC__, TYPE >()(acctmp);
                        typename FUN::template FinalizeOutput< __TYPEACC__, TYPE >()(acctmp, out + i * DIMOUT, i);
                        }
                        
                        // N.B.: In the following code, we assume that the x-ranges do not overlap.
                        //       Otherwise, we'd have to assume that DIMRED == DIMOUT
                        //       or allocate a buffer of size nx * DIMRED. This may be done in the future.
                        // Cf. reduction.h: 
                        //    FUN::tagJ = 1 for a reduction over j, result indexed by i
                        //    FUN::tagJ = 0 for a reduction over i, result indexed by j
                        
                        int nranges = FUN::tagJ ? nranges_x : nranges_y;
                        __INDEX__* ranges_x = FUN::tagJ ? ranges[0] : ranges[3];
                        __INDEX__* slices_x = FUN::tagJ ? ranges[1] : ranges[4];
                        __INDEX__* ranges_y = FUN::tagJ ? ranges[2] : ranges[5];
                        
                        int indices_i[SIZEI], indices_j[SIZEJ], indices_p[SIZEP];  // Buffers for the "broadcasted indices"
                        for (int k = 0; k < SIZEI; k++) { indices_i[k] = 0; }  // Fill the "offsets" with zeroes,
                        for (int k = 0; k < SIZEJ; k++) { indices_j[k] = 0; }  // the default value when nbatchdims == 0.
                        for (int k = 0; k < SIZEP; k++) { indices_p[k] = 0; }
                        
                        for (int range_index = 0; range_index < nranges; range_index++) {
                        
                        __INDEX__ start_x = ranges_x[2 * range_index];
                        __INDEX__ end_x = ranges_x[2 * range_index + 1];
                        
                        __INDEX__ start_slice = (range_index < 1) ? 0 : slices_x[range_index - 1];
                        __INDEX__ end_slice = slices_x[range_index];
                        
                        // If needed, compute the "true" start indices of the range, turning
                        // the "abstract" index start_x into an array of actual "pointers/offsets" stored in indices_i:
                        if (nbatchdims > 0) {
                        vect_broadcast_index(start_x, nbatchdims, SIZEI, shapes, shapes_i, indices_i);
                        // And for the parameters, too:
                        vect_broadcast_index(range_index, nbatchdims, SIZEP, shapes, shapes_p, indices_p);
                        load< DIMSP, INDSP >(0, pp, args, indices_p); // Load the paramaters, once per tile
                        }
                        
                        #pragma omp parallel for   
                        for (__INDEX__ i = start_x; i < end_x; i++) {
                        TYPE xi[DIMX], yj[DIMY], fout[DIMFOUT];
                        __TYPEACC__ acc[DIMRED];
                        #if SUM_SCHEME == BLOCK_SUM
                        // additional tmp vector to store intermediate results from each block
                        TYPE tmp[DIMRED];
                        #elif SUM_SCHEME == KAHAN_SCHEME
                        // additional tmp vector to accumulate errors
                        const int DIM_KAHAN = FUN::template KahanScheme<__TYPEACC__,TYPE>::DIMACC;
                        TYPE tmp[DIM_KAHAN];
                        #endif
                        if (nbatchdims == 0) {
                        load< DIMSX, INDSI >(i, xi, args);
                        } else {
                        load< DIMSX, INDSI >(i - start_x, xi, args, indices_i);
                        }
                        typename FUN::template InitializeReduction< __TYPEACC__, TYPE >()(acc);   // tmp = 0
                        #if SUM_SCHEME == BLOCK_SUM
                        typename FUN::template InitializeReduction< TYPE, TYPE >()(tmp);   // tmp = 0
                        #elif SUM_SCHEME == KAHAN_SCHEME
                        VectAssign<DIM_KAHAN>(tmp,0.0f);
                        #endif
                        for (__INDEX__ slice = start_slice; slice < end_slice; slice++) {
                        __INDEX__ start_y = ranges_y[2 * slice];
                        __INDEX__ end_y = ranges_y[2 * slice + 1];
                        
                        // If needed, compute the "true" start indices of the range, turning
                        // the "abstract" index start_y into an array of actual "pointers/offsets" stored in indices_j:
                        if (nbatchdims > 0) {
                        vect_broadcast_index(start_y, nbatchdims, SIZEJ, shapes, shapes_j, indices_j);
                        }
                        
                        if (nbatchdims == 0) {
                        for (int j = start_y; j < end_y; j++) {
                        load< DIMSY, INDSJ >(j, yj, args);
                        call< DIMSX, DIMSY, DIMSP >(fun, fout, xi, yj, pp);
                        #if SUM_SCHEME == BLOCK_SUM
                        typename FUN::template ReducePairShort< TYPE, TYPE >()(tmp, fout, j); // tmp += fout
                        if ((j+1)%200) {
                        typename FUN::template ReducePair< __TYPEACC__, TYPE >()(acc, tmp); // acc += tmp
                        typename FUN::template InitializeReduction< TYPE, TYPE >()(tmp);   // tmp = 0
                        }
                        #elif SUM_SCHEME == KAHAN_SCHEME
                        typename FUN::template KahanScheme<__TYPEACC__,TYPE>()(acc, fout, tmp);
                        #else
                        typename FUN::template ReducePairShort< __TYPEACC__, TYPE >()(acc, fout, j); // acc += fout
                        #endif
                        }
                        }
                        else {
                        for (int j = start_y; j < end_y; j++) {
                        load< DIMSY, INDSJ >(j - start_y, yj, args, indices_j);
                        call< DIMSX, DIMSY, DIMSP >(fun, fout, xi, yj, pp);
                        #if SUM_SCHEME == BLOCK_SUM
                        typename FUN::template ReducePairShort< TYPE, TYPE >()(tmp, fout, j - start_y); // tmp += fout
                        if ((j+1)%200) {
                        typename FUN::template ReducePair< __TYPEACC__, TYPE >()(acc, tmp); // acc += tmp
                        typename FUN::template InitializeReduction< TYPE, TYPE >()(tmp);   // tmp = 0
                        }
                        #elif SUM_SCHEME == KAHAN_SCHEME
                        typename FUN::template KahanScheme<__TYPEACC__,TYPE>()(acc, fout, tmp);
                        #else
                        typename FUN::template ReducePairShort< __TYPEACC__, TYPE >()(acc, fout, j - start_y); // acc += fout
                        #endif
                        }
                        }
                        }
                        #if SUM_SCHEME == BLOCK_SUM
                        typename FUN::template ReducePair< __TYPEACC__, TYPE >()(acc, tmp); // acc += tmp
                        #endif
                        typename FUN::template FinalizeOutput< __TYPEACC__, TYPE >()(acc, out + i * DIMOUT, i);
                        }
                        
                        }
                        
                        return 0;
                        }
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
















