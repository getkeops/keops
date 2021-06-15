from keops.python_engine.formulas.Operation import Broadcast
from keops.python_engine.formulas.VectorizedScalarOp import VectorizedScalarOp
from keops.python_engine.formulas.variables.Zero import Zero
from keops.python_engine.formulas.vectOps.Scalprod import Scalprod
from keops.python_engine.formulas.maths.Square import Square
##########################
######    Divide     #####
##########################



class Divide(VectorizedScalarOp):
    """the binary divide operation"""
    string_id = "Divide"
    print_spec = "/", "mid", 3

    def __new__(cls, arg0, arg1):
        if isinstance(arg0, Zero):
            return Broadcast(arg0, arg1.dim)
        elif isinstance(arg1, Zero):
            raise ValueError("division by zero")
        elif isinstance(arg1, int):
            from keops.python_engine.formulas.variables.IntCst import IntCst
            return Divide(arg0, IntCst(arg1))
        else:
            return super(Divide, cls).__new__(cls)

    def ScalarOp(self, out, arg0, arg1):
        """returns the atomic piece of c++ code to evaluate the function on arg and return the result in out"""
        return f"{out.id} = {arg0.id} / {arg1.id};\n"

    #  \diff_V (A/B) = ((\diff_V A) * B - A * (\diff_V B)) / B^2
    def DiffT(self, v, gradin):
        fa, fb = self.children
        if fa.dim == 1 and fb.dim > 1:
            return (fa.Grad(v, Scalprod(gradin, fb)) - fb.Grad(v, fa * gradin)) / Square(fb)
        elif fb.dim == 1 and fa.dim > 1:
            return (fa.Grad(v, fb * gradin) - fb.Grad(v, Scalprod(gradin, fa))) / Square(fb)
        else:
            return (fa.Grad(v, fb * gradin) - fb.Grad(v, fa * gradin)) / Square(fb)
