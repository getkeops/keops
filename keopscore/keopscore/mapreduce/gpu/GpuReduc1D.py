from keopscore.binders.nvrtc.Gpu_link_compile import Gpu_link_compile
from keopscore.utils.meta_toolbox.c_expression import c_pointer
from keopscore.utils.meta_toolbox.c_function import cuda_global_kernel
from keopscore.utils.meta_toolbox.c_code import c_code
from keopscore.utils.meta_toolbox.c_instruction import (
    c_instruction,
    c_comment,
    c_instruction_from_string,
)
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
            dtype,
            varloader.dimy_local,
            f"({yj} + threadIdx.x * {varloader.dimy_local})",
        )
        yjrel = c_array(dtype, varloader.dimy_local, "yjrel")

        j_start = c_variable("signed long int", "jstart")
        j_rel = c_variable("signed long int", "jrel")
        j_call = j_start + j_rel
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

        sync_threads = c_instruction_from_string("__syncthreads()")

        out = c_variable(c_pointer(dtype), "out")

        def cond_i(*instructions):
            return c_if(i < nx, instructions)

        def cond_j(*instructions):
            return c_if(j < ny, instructions)

        # print(red_formula.formula(fout, table, i, jreltile, tagI))
        # exit()

        code = self.headers + cuda_global_kernel(
            "GpuConv1DOnDevice",
            (nx, ny, out, arg),
            (
                i.declare_assign(blockIdx_x * blockDim_x + threadIdx_x),
                yj.declare(),
                param_loc.declare(),
                varloader.load_vars("p", param_loc, args),
                fout.declare(),
                xi.declare(),
                acc.declare(),
                sum_scheme.declare_temporary_accumulator(),
                cond_i(
                    red_formula.InitializeReduction(acc),
                    sum_scheme.initialize_temporary_accumulator_first_init(),
                    varloader.load_vars("i", xi, args, row_index=i),
                ),
                c_for(
                    (jstart.declare_assign(0), tile.assign(0)),
                    jstart < ny,
                    (jstart.add_assign(blockDim_x), tile.plus_plus),
                    (
                        j.declare_assign(tile * blockDim_x + threadIdx_x),
                        cond_j(varloader.load_vars("j", yjloc, args, row_index=j)),
                        sync_threads,
                        cond_i(
                            yjrel.c_var.declare_assign(yj.c_var),
                            sum_scheme.initialize_temporary_accumulator_block_init(),
                            c_for(
                                init=jrel.declare_assign(0),
                                end=(jrel < blockDim_x).logical_and(jrel < ny - jstart),
                                loop=(
                                    jrel.plus_plus,
                                    yjrel.c_var.add_assign(varloader.dimy_local),
                                ),
                                body=(
                                    red_formula.formula(fout, table, i, jreltile, tagI),
                                    sum_scheme.accumulate_result(acc, fout, jreltile),
                                ),
                            ),
                            sum_scheme.final_operation(acc),
                        ),
                        sync_threads,
                    ),
                ),
                cond_i(red_formula.FinalizeOutput(acc, outi, i)),
            ),
        )

        self.code = str(code)
