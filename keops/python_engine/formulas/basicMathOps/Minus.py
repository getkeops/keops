from keops.python_engine.formulas.VectorizedScalarOp import VectorizedScalarOp
from keops.python_engine.formulas.Zero import Zero


##########################
######    Minus      #####
##########################

class Minus_(VectorizedScalarOp):
    """the "minus" vectorized operation"""
    string_id = "Minus"
    print_spec = "-", "pre", 2

    def ScalarOp(self, out, arg):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return f"{out.id} = -{arg.id};\n"

    def DiffT(self, v, gradin):
        f = self.children[0]
        return -f.Grad(v, gradin)


def Minus(arg):
    if isinstance(arg, Zero):
        return arg
    else:
        return Minus_(arg)