from keopscore.binders.nvrtc.Gpu_link_compile import Gpu_link_compile
from keopscore.mapreduce.gpu.GpuAssignZero import GpuAssignZero
from keopscore.mapreduce.MapReduce import MapReduce
from keopscore.utils.code_gen_utils import (
    c_if,
    c_variable,
    c_array,
)


class GpuReduc1D(MapReduce, Gpu_link_compile):
    # class for generating the final C++ code, Gpu version

    AssignZero = GpuAssignZero
    force_all_local = False

    def __init__(self, *args):
        MapReduce.__init__(self, *args)
        Gpu_link_compile.__init__(self)
        self.dimy = self.varloader.dimy

    def get_code(self):
        super().get_code()

        red_formula = self.red_formula
        tagI = red_formula.tagI
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
        yj = c_array(f"extern __shared__ {dtype}", "", "yj")
        yjloc = c_array(
            dtype, varloader.dimy_local, f"({yj} + threadIdx.x * {varloader.dimy_local})"
        )
        yjrel = c_array(dtype, varloader.dimy_local, "yjrel")

        j_start = c_variable("signed long int", "jstart")
        j_rel = c_variable("signed long int", "jrel")
        j_call = j_start+j_rel
        table = varloader.table(self.xi, yjrel, self.param_loc, args, i, j_call)
        
        

        nx = c_variable("signed long int", "nx")
        ny = c_variable("signed long int", "ny")

        blockIdx_x = c_variable("int", "blockIdx.x")
        blockDim_x = c_variable("int", "blockDim.x")
        threadIdx_x = c_variable("int", "threadIdx.x")

        tile = c_variable("signed long int", "tile")
        jstart = c_variable("signed long int", "jstart")

        jrel = c_variable("signed long int", "jrel")
        jreltile = jrel + tile * blockDim_x

        self.code = f"""
                          
                        {self.headers}
                        
                        extern "C" __global__ void GpuConv1DOnDevice(signed long int {nx}, signed long int {ny}, {dtype} *out, {dtype} **{arg.id}) {{
    
                          // get the index of the current thread
                          {i.declare_assign(blockIdx_x * blockDim_x + threadIdx_x)}

                          // declare shared mem
                          {yj.declare()}

                          // load parameters variables from global memory to local thread memory
                          {param_loc.declare()}
                          {varloader.load_vars("p", param_loc, args)}

                          {fout.declare()}
                          {xi.declare()}
                          {acc.declare()}
                          {sum_scheme.declare_temporary_accumulator()}

                          {c_if(i<nx,
                                red_formula.InitializeReduction(acc),
                                sum_scheme.initialize_temporary_accumulator_first_init(),
                                varloader.load_vars('i', xi, args, row_index=i))}

                          for (signed long int {jstart} = 0, {tile} = 0; {jstart<ny}; {jstart} += {blockDim_x}, {tile}++) {{

                            // get the current column
                            {j.declare_assign(tile * blockDim_x + threadIdx_x)}

                            {c_if(j<ny, 
                                  varloader.load_vars("j",yjloc, args, row_index=j), 
                                  comment="we load yj from device global memory only if j<ny")}
                                  
                            __syncthreads();

                            if ({i<nx}) {{ // we compute x1i only if needed
                              {dtype} * {yjrel} = {yj};
                              {sum_scheme.initialize_temporary_accumulator_block_init()}
                              for (signed long int {jrel} = 0; ({jrel<blockDim_x}) && ({jrel<ny-jstart}); {jrel}++, {yjrel} += {varloader.dimy_local}) {{
                                {red_formula.formula(fout, table, i, jreltile, tagI)} // Call the function, which outputs results in fout
                                {sum_scheme.accumulate_result(acc, fout, jreltile)}
                              }}
                              {sum_scheme.final_operation(acc)}
                            }}
                            __syncthreads();
                          }}
                          
                          {c_if(i<nx, red_formula.FinalizeOutput(acc, outi, i))}

                        }}
                    """
