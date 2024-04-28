import keopscore
from keopscore.utils.meta_toolbox import (
    VectApply,
    c_for_loop,
    c_comment,
    c_empty_instruction,
    c_fixed_size_array,
)
from keopscore.formulas.Operation import Operation
from keopscore.utils.misc_utils import KeOps_Error


class VectorizedScalarOp(Operation):
    # class for operations that are vectorized or broadcasted
    # scalar operations,
    # such as Exp(f), Cos(f), Mult(f,g), Subtract(f,g), etc.

    def __init__(self, *args, params=()):
        dims = set(arg.dim for arg in args)
        if len(dims) > 2 or (len(dims) == 2 and min(dims) != 1):
            KeOps_Error("dimensions are not compatible for VectorizedScalarOp")
        super().__init__(*args, params=params)

    @property
    def dim(self):
        # dim gives the output dimension of the operation,
        # here it is the same as the output dimension of the child operation
        return max(child.dim for child in self.children)

    def get_code_and_expr_elem(self, dtype, table, i, j, tagI, elem):
        # Evaluation of the child operations
        if len(self.children) > 0:
            code_args, code_args_elem, args = zip(
                *(
                    child.get_code_and_expr_elem(
                        dtype, table, i, j, tagI, 0 if child.dim == 1 else elem
                    )
                    for child in self.children
                )
            )
            code = sum(code_args, c_empty_instruction)
            code_elem = sum(code_args_elem, c_empty_instruction)
        else:
            args, code, code_elem = (), c_empty_instruction, c_empty_instruction
        # Finally, evaluation of the operation itself
        if hasattr(self, "ScalarOpFun"):
            out = type(self).ScalarOpFun(*args, *self.params)
        else:
            out = self.get_out_var(dtype)
            code_elem += out.declare() + self.ScalarOp(out, *args)
        return code, code_elem, out

    def __call__(self, out, table, i=None, j=None, tagI=None):
        code = c_comment(f"Starting code block for {self.__repr__()}")
        forloop, k = c_for_loop(0, self.dim, 1, pragma_unroll=True)
        code_k, code_elem_k, out_k_expr = self.get_code_and_expr_elem(
            out.dtype, table, i, j, tagI, k
        )
        code += code_k + forloop(code_elem_k + out[k].assign(out_k_expr))
        code += c_comment(f"Finished code block for {self.__repr__()}")
        return code

    def Op(self, out, table, *args):
        # Atomic evaluation of the operation : it consists in a simple
        # for loop around the call to the correponding scalar operation
        return VectApply(self.ScalarOp, out, *args)

    def ScalarOp(self, out, *args):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return out.assign(type(self).ScalarOpFun(*args, *self.params), out.dtype)

    def DiffT(self, v, gradin):
        derivatives = self.Derivative(*self.children)
        if len(self.children) == 1:
            derivatives = (derivatives,)
        return sum(
            (f.DiffT(v, gradin) * df if df.dim == 1 else f.DiffT(v, gradin * df))
            for f, df in zip(self.children, derivatives)
        )

    def Derivative(self):
        pass

    @property
    def is_chunkable(self):
        return all(child.is_chunkable for child in self.children)

    def chunked_version(self, dimchk):
        return type(self)(
            *(
                (child if child.dim == 1 else child.chunked_version(dimchk))
                for child in self.children
            )
        )

    def chunked_vars(self, cat):
        res = set()
        for child in self.children:
            if child.dim != 1:
                res = set.union(res, set(child.chunked_vars(cat)))
        return list(res)

    def notchunked_vars(self, cat):
        res = set()
        for child in self.children:
            if child.dim == 1:
                res = set.union(res, set(child.Vars(cat)))
            else:
                res = set.union(res, set(child.notchunked_vars(cat)))
        return list(res)

    enable_test = True
