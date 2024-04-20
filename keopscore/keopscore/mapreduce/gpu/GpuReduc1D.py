from keopscore.binders.nvrtc.Gpu_link_compile import Gpu_link_compile
from keopscore.utils.meta_toolbox.c_expression import c_pointer
from keopscore.utils.meta_toolbox.c_function import cuda_global_kernel
from keopscore.utils.meta_toolbox.c_code import c_code
from keopscore.utils.meta_toolbox.c_instruction import c_instruction
from keopscore.utils.meta_toolbox.c_for import c_for
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

        blockIdx_x = cuda_global_kernel.blockIdx_x
        blockDim_x = cuda_global_kernel.blockDim_x
        threadIdx_x = cuda_global_kernel.threadIdx_x

        tile = c_variable("signed long int", "tile")
        jstart = c_variable("signed long int", "jstart")

        jrel = c_variable("signed long int", "jrel")
        jreltile = jrel + tile * blockDim_x

        sync_threads = c_instruction("__syncthreads()", set(), set())

        out = c_variable(c_pointer(dtype),"out")

        code = self.headers + cuda_global_kernel(
            "GpuConv1DOnDevice",
            (nx,ny,out,arg),
            (
                i.declare_assign(blockIdx_x * blockDim_x + threadIdx_x, comment = "get the index of the current thread"),
                yj.declare(comment="declare shared mem"),
                param_loc.declare(comment="load parameters variables from global memory to local thread memory"),
                varloader.load_vars("p", param_loc, args),
                fout.declare(),
                xi.declare(),
                acc.declare(),
                sum_scheme.declare_temporary_accumulator(),
                c_if(
                    i<nx,
                    (
                        red_formula.InitializeReduction(acc),
                        sum_scheme.initialize_temporary_accumulator_first_init(),
                        varloader.load_vars('i', xi, args, row_index=i),
                    )
                ),
                c_for(
                    (jstart.declare_assign(0),tile.assign(0)),
                    jstart<ny,
                    (jstart.add_assign(blockDim_x),tile.plus_plus),
                    (
                        j.declare_assign(tile * blockDim_x + threadIdx_x, comment="get the current column"),
                        c_if(
                            j<ny, 
                            varloader.load_vars("j",yjloc, args, row_index=j), 
                            comment="we load yj from device global memory only if j<ny"
                        ),
                        sync_threads,
                        c_if(
                            i<nx,
                            (
                                yjrel.c_var.declare_assign(yj.c_var),
                                sum_scheme.initialize_temporary_accumulator_block_init(),
                                c_for(
                                    jrel.declare_assign(0),
                                    (jrel<blockDim_x).logical_and(jrel<ny-jstart),
                                    (jrel.plus_plus, yjrel.c_var.add_assign(varloader.dimy_local)),
                                    (
                                        red_formula.formula(fout, table, i, jreltile, tagI),
                                        sum_scheme.accumulate_result(acc, fout, jreltile)
                                    )
                                ),
                               sum_scheme.final_operation(acc)
                            ),
                            comment = "we compute x1i only if needed"
                        ),
                        sync_threads
                    )
                ),
                c_if(i<nx, red_formula.FinalizeOutput(acc, outi, i))
            )
        )

        self.code = str(code)

        f = open("ess.cu","w")
        f.write(self.code)
        f.close()


