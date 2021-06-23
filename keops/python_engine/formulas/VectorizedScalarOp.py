from keops.python_engine.utils.code_gen_utils import VectApply
from keops.python_engine.formulas.Operation import Operation


class VectorizedScalarOp(Operation):
    # class for operations that are vectorized or broadcasted
    # scalar operations,
    # such as Exp(f), Cos(f), Mult(f,g), Subtract(f,g), etc.

    def __init__(self, *args, params=()):
        dims = set(arg.dim for arg in args)
        if len(dims)>2 or (len(dims)==2 and min(dims)!=1):
            raise ValueError("dimensions are not compatible for VectorizedScalarOp")
        super().__init__(*args, params=params)

    @property
    def dim(self):
        # dim gives the output dimension of the operation,
        # here it is the same as the output dimension of the child operation
        return max(child.dim for child in self.children)

    def Op(self, out, table, *args):
        # Atomic evaluation of the operation : it consists in a simple
        # for loop around the call to the correponding scalar operation
        return VectApply(self.ScalarOp, out, *args)
    
    def ScalarOp(self, out, *args):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return out.assign(type(self).ScalarOpFun(*args, *self.params))
        
    def DiffT(self, v, gradin):        
        derivatives = self.Derivative(*self.children, *self.params)
        if len(self.children)==1:
            derivatives = (derivatives,)
        return sum(f.DiffT(v,gradin*df) for f,df in zip(self.children, derivatives))
