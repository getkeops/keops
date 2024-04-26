from keopscore.formulas.reductions import *
from keopscore.formulas.GetReduction import GetReduction
from keopscore.utils.code_gen_utils import Var_loader
from keopscore.utils.meta_toolbox import (
    c_pointer_dtype,
    c_expression_from_string,
    c_array_from_address,
    new_c_name,
    c_include,
    c_define,
)


class MapReduce:
    """
    base class for map-reduce schemes
    """

    def __init__(
        self,
        red_formula_string,
        aliases,
        nargs,
        dtype,
        dtypeacc,
        sum_scheme_string,
        tagHostDevice,
        tagCpuGpu,
        tag1D2D,
        use_half,
        use_fast_math,
        device_id,
    ):
        self.red_formula_string = red_formula_string
        self.aliases = aliases

        self.red_formula = GetReduction(red_formula_string, aliases=aliases)

        self.dtype = dtype
        self.dtypeacc = dtypeacc
        self.nargs = nargs
        self.sum_scheme_string = sum_scheme_string
        self.tagHostDevice, self.tagCpuGpu, self.tag1D2D = (
            tagHostDevice,
            tagCpuGpu,
            tag1D2D,
        )
        self.use_half = use_half
        self.use_fast_math = use_fast_math
        self.device_id = device_id
        self.varloader = Var_loader(
            self.red_formula, force_all_local=self.force_all_local
        )

    def get_code(self):
        self.headers = c_define("C_CONTIGUOUS", "1")

        if self.use_half == 1:
            self.headers += c_define("USE_HALF", "1")
            self.headers += c_define("half2", "__half2")
            self.headers += c_include("cuda_fp16.h")
        else:
            self.headers += c_define("USE_HALF", "0")

        red_formula = self.red_formula
        formula = red_formula.formula
        dtype = self.dtype
        dtypeacc = self.dtypeacc
        nargs = self.nargs
        self.sum_scheme = eval(self.sum_scheme_string)(red_formula, dtype)

        self.i = i = c_variable("signed long int", "i")
        self.j = j = c_variable("signed long int", "j")

        nx = c_variable("signed long int", "nx")
        ny = c_variable("signed long int", "ny")

        self.xi = c_fixed_size_array(dtype, self.varloader.dimx_local, "xi")
        self.param_loc = c_fixed_size_array(
            dtype, self.varloader.dimp_local, "param_loc"
        )

        argname = new_c_name("arg")
        self.arg = c_variable(c_pointer_dtype(c_pointer_dtype(dtype)), argname)
        self.args = [self.arg[k] for k in range(nargs)]

        self.acc = c_fixed_size_array(dtypeacc, red_formula.dimred, "acc")
        self.acctmp = c_fixed_size_array(dtypeacc, red_formula.dimred, "acctmp")
        self.fout = c_fixed_size_array(dtype, formula.dim, "fout")
        self.out = c_variable(c_pointer_dtype(dtype), "out")
        self.outi = c_array_from_address(
            red_formula.dim, self.out + i * red_formula.dim
        )
