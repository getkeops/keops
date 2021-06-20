from keops.python_engine.formulas.VectorizedScalarOp import VectorizedScalarOp

from keops.python_engine.formulas.variables.Zero import Zero


##########################
######    Square     #####
##########################


class Square(VectorizedScalarOp):
    """the square vectorized operation"""
    string_id = "Square"
    print_spec = "**2", "post", 1

    def __new__(cls, arg):
        if isinstance(arg, Zero):
            return arg
        else:
            return super(Square, cls).__new__(cls)

    def ScalarOp(self, out, arg):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return out.assign(arg*arg)

    def DiffT(self, v, gradin):
        from keops.python_engine.formulas.variables.IntCst import IntCst
        # [\partial_V (F)**2].gradin = F * [\partial_V F].gradin
        f = self.children[0]
        return IntCst(2) * f.Grad(v, f * gradin)
