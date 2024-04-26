import keopscore
from keopscore.utils.meta_toolbox.c_array import c_fixed_size_array
from keopscore.utils.meta_toolbox import (
    VectApply,
    c_for_loop,
    c_comment,
    c_empty_instruction,
)
from keopscore.formulas.Operation import Operation
from keopscore.formulas.Chunkable_Op import Chunkable_Op
from keopscore.utils.misc_utils import KeOps_Error
from keopscore.utils.meta_toolbox import c_zero_float, VectApply


class InnerReductionOp(Chunkable_Op):
    # class for operations that represnts reductions
    # such as Sum(f), Scalprod(f)

    dim = 1

    def Op(self, out, table, *args):
        return out.assign(c_zero_float) + VectApply(self.ScalarOp, out, *args)

    def __call__(self, out, table, i=None, j=None, tagI=None):
        code = c_comment(f"Starting code block for {self.__repr__()}")
        forloop, k = c_for_loop(0, self.children[0].dim, 1, pragma_unroll=True)
        if len(self.children) > 0:
            code_args, code_args_elem, args = zip(
                *(
                    child.get_code_and_expr_elem(out.dtype, table, i, j, tagI, k)
                    for child in self.children
                )
            )
            code += sum(code_args, c_empty_instruction)
            code_elem = sum(code_args_elem, c_empty_instruction)
        else:
            args, code_elem = (), c_empty_instruction
        out_value = out[0] if isinstance(out, c_fixed_size_array) else out
        code += out.assign(c_zero_float) + forloop(
            code_elem + self.ScalarOp(out_value, *args)
        )
        code += c_comment(f"Finished code block for {self.__repr__()}")
        return code
