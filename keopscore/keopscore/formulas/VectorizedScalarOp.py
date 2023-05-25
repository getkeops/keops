from keopscore.utils.code_gen_utils import VectApply
from keopscore.formulas.Operation import Operation
from keopscore.utils.misc_utils import KeOps_Error


class VectorizedScalarOp(Operation):
    # class for operations that are vectorized or broadcasted
    # scalar operations,
    # such as Exp(f), Cos(f), Mult(f,g), Subtract(f,g), etc.

    def __init__(self, *args, params=()):
        shapes = tuple(arg.shape for arg in args)
        if len(shapes) > 2 or (len(shapes) == 2 and not all((x==y or min((x,y))==1) for (x,y) in zip(*shapes))):
            KeOps_Error("dimensions are not compatible for VectorizedScalarOp")
        super().__init__(*args, params=params)

    @property
    def shape(self):
        # shape gives the output shape of the operation,
        # here we use broadcasting rules to infer the new shape
        return tuple(max(z) for z in zip(*(child.shape for child in self.children)))

    def Op(self, out, table, *args):
        # Atomic evaluation of the operation : it consists in a simple
        # for loop around the call to the correponding scalar operation
        print()
        print("out=", out)
        print("args[0]=", args[0])
        input()
        return VectApply(self.ScalarOp, out, *args)

    def ScalarOp(self, out, *args):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return out.assign(type(self).ScalarOpFun(*args, *self.params))

    def DiffT(self, v, gradin):
        derivatives = self.Derivative(*self.children, *self.params)
        if len(self.children) == 1:
            derivatives = (derivatives,)
        return sum(f.DiffT(v, gradin * df) for f, df in zip(self.children, derivatives))

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
            ),
            *self.params
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
