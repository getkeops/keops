from keopscore.binders.nvrtc.Gpu_link_compile import Gpu_link_compile
from keopscore.mapreduce.gpu.GpuAssignZero import GpuAssignZero
from keopscore.mapreduce.MapReduce import MapReduce
from keopscore.utils.code_gen_utils import (
    c_variable,
    c_array,
)


class GpuReduc1D(MapReduce, Gpu_link_compile):
    # class for generating the final C++ code, Gpu version

    AssignZero = GpuAssignZero

    def __init__(self, *args):
        MapReduce.__init__(self, *args)
        Gpu_link_compile.__init__(self)
        self.dimy = self.varloader.dimy

    def get_code(self):
        super().get_code()

        red_formula = self.red_formula
        dtype = self.dtype
        varloader = self.varloader

        i = self.i
        j = self.j
        fout = self.fout
        outi = self.outi
        acc = self.acc
        arg = self.arg
        args = self.args
        sum_scheme = self.sum_scheme

        param_loc = self.param_loc
        xi = self.xi
        yjloc = c_array(dtype, varloader.dimy, f"(yj + threadIdx.x * {varloader.dimy})")
        yjrel = c_array(dtype, varloader.dimy, "yjrel")
        table = varloader.table(self.xi, yjrel, self.param_loc)
        jreltile = c_variable("int", "(jrel + tile * blockDim.x)")

        self.code = f"""
                          
                        {self.headers}
                        
                        extern "C" __global__ void GpuConv1DOnDevice(int nx, int ny, {dtype} *out, {dtype} **{arg.id}) {{
    
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
                                {red_formula.formula(fout, table)} // Call the function, which outputs results in fout
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
                    """
