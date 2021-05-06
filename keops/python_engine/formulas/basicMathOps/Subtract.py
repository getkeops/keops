from keops.python_engine.formulas.Operation import Broadcast
from keops.python_engine.formulas.VectorizedScalarOp import VectorizedScalarOp
from keops.python_engine.formulas.Zero import Zero


##########################
######    Subtract   #####
##########################

class Subtract_(VectorizedScalarOp):
    """the binary subtract operation"""
    string_id = "Subtract"
    print_spec = "-", "mid", 4

    def ScalarOp(self, out, arg0, arg1):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return f"{out.id} = {arg0.id}-{arg1.id};\n"

    def DiffT(self, v, gradin):
        fa, fb = self.children
        return fa.Grad(v, gradin) - fb.Grad(v, gradin)


def Subtract(arg0, arg1):
    if isinstance(arg0, Zero):
        return -Broadcast(arg1, arg0.dim)
    elif isinstance(arg1, Zero):
        return Broadcast(arg0, arg1.dim)
    else:
        return Subtract_(arg0, arg1)
