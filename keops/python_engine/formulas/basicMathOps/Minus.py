from keops.python_engine.formulas.VectorizedScalarOp import VectorizedScalarOp
from keops.python_engine.formulas.variables.Zero import Zero


##########################
######    Minus      #####
##########################

class Minus(VectorizedScalarOp):
    """the "minus" vectorized operation"""
    string_id = "Minus"
    print_spec = "-", "pre", 2

    def __new__(cls, arg):
        if isinstance(arg, Zero):
            return arg
        else:
            return super(Minus, cls).__new__(cls)

    def ScalarOp(self, out, arg):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return f"{out.id} = -{arg.id};\n"

    def DiffT(self, v, gradin):
        f = self.children[0]
        return -f.Grad(v, gradin)
