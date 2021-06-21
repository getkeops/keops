from keops.python_engine.formulas.VectorizedScalarOp import VectorizedScalarOp

from keops.python_engine.formulas.variables.Zero import Zero


##########################
######    Square     #####
##########################


class Square_Impl(VectorizedScalarOp):
    """the square vectorized operation"""
    string_id = "Square"
    print_spec = "**2", "post", 1

    def ScalarOp(self, out, arg):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return out.assign(arg*arg)

    def DiffT(self, v, gradin):
        from keops.python_engine.formulas.variables.IntCst import IntCst
        # [\partial_V (F)**2].gradin = F * [\partial_V F].gradin
        f = self.children[0]
        return IntCst(2) * f.Grad(v, f * gradin)


# N.B. The following separate function should theoretically be implemented
# as a __new__ method of the previous class, but this can generate infinite recursion problems
def Square(arg):
    if isinstance(arg, Zero):
        return arg
    else:
        return Square_Impl(arg)

