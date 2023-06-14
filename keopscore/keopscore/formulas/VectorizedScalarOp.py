from keopscore.utils.code_gen_utils import VectApply
from keopscore.formulas.Operation import Operation
from keopscore.utils.misc_utils import KeOps_Error
from keopscore.formulas.Operation import BroadcastT

def broadcast_shapes(shapes):
    # check that input shapes are compatible for broadcasting
    # and return the output broadcasted shape.
    # for example if shapes[0]=(2,3,1) and shapes[1]=(2,1,4), then
    # get_shape(shapes) will return (2,3,4)
    n = len(shapes)
    ndims = list(len(shape) for shape in shapes)
    ndim = max(ndims)
    for i in range(n):
        shapes[i] = shapes[i] + (1,)*(ndim-ndims[i])
    shapeout = []
    for k in range(ndim):
        dims = set(shape[k] for shape in shapes)
        if len(dims) == 2 and min(dims) == 1:
            dimout = max(dims)
        elif len(dims) == 1:
            dimout = dims.pop()
        else:
            raise ValueError(f"Incompatible shapes for broadcasting. The axis dimensions at non-singleton dimension {k} are {', '.join(list(str(shape[k]) for shape in shapes))}.")
        shapeout.append(dimout)
    return tuple(shapeout)

class VectorizedScalarOp(Operation):
    # class for operations that are vectorized or broadcasted
    # scalar operations,
    # such as Exp(f), Cos(f), Mult(f,g), Subtract(f,g), etc.

    def check_shapes(self):
        if self.shapes is None:
            # here shapes of args, are not provided, so args are considered vectors.
            # So we check basic broadcast rule : inputs all have same dimensions, or some are scalars.
            dims = set(arg.dim for arg in self.children)
            if len(dims) > 2 or (len(dims) == 2 and min(dims) != 1):
                KeOps_Error("dimensions are not compatible for VectorizedScalarOp")
        else:
            # N.B. we should check shapes are compatible for broadcast here.
            # But currently this mode with 'shapes' parameter is only used with SymbolicTensor class from pykeops,
            # which already performs its own compatibility check.
            pass

    @property
    def dim(self):
        # dim gives the output dimension of the operation,
        # here it is the same as the output dimension of the child operation
        return max(child.dim for child in self.children)

    def Op(self, out, table, *args):
        # Atomic evaluation of the operation : it consists in a simple
        # for loop around the call to the correponding scalar operation
        return VectApply(self.ScalarOp, out, *args, shapes=self.shapes)
    
    scalar_op_params = ()

    def ScalarOp(self, out, *args):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return out.assign(type(self).ScalarOpFun(*args, *self.scalar_op_params))

    def DiffT(self, v, gradin):
        derivatives = self.Derivative(*self.children, *self.params)
        if len(self.children) == 1:
            derivatives = (derivatives,)
        return sum(f.DiffT(v, BroadcastT(gradin,f.dim) * df) for f, df in zip(self.children, derivatives))

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
