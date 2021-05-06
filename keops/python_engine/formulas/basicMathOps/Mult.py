from keops.python_engine.formulas.Operation import VectorizedScalarOp, Zero, Broadcast, IntCst
from keops.python_engine.formulas.basicMathOps import Scalprod

##########################
######    Mult       #####
##########################



class Mult_(VectorizedScalarOp):
    """the binary multiply operation"""
    string_id = "Mult"
    print_spec = "*", "mid", 3

    def ScalarOp(self, out, arg0, arg1):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return f"{out.id} = {arg0.id}*{arg1.id};\n"

    #  \diff_V (A*B) = (\diff_V A) * B + A * (\diff_V B)
    def DiffT(self, v, gradin):
        fa, fb = self.children
        if fa.dim == 1 and fb.dim > 1:
            return fa.Grad(v, Scalprod(gradin, fb)) + fb.Grad(v, fa * gradin)
        elif fb.dim == 1 and fa.dim > 1:
            return fa.Grad(v, fb * gradin) + fb.Grad(v, Scalprod(gradin, fa))
        else:
            return fa.Grad(v, fb * gradin) + fb.Grad(v, fa * gradin)


def Mult(arg0, arg1):
    if isinstance(arg0, Zero):
        return Broadcast(arg0, arg1.dim)
    elif isinstance(arg1, Zero):
        return Broadcast(arg1, arg0.dim)
    elif isinstance(arg1, int):
        return Mult(IntCst(arg1), arg0)
    else:
        return Mult_(arg0, arg1)
