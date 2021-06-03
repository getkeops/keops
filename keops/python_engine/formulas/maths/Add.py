from keops.python_engine.formulas.maths.Operation import Broadcast
from keops.python_engine.formulas.maths.VectorizedScalarOp import VectorizedScalarOp
from keops.python_engine.formulas.variables.Zero import Zero


##########################
######    Add        #####
##########################

class Add(VectorizedScalarOp):
    """the binary addition operation"""
    string_id = "Add"
    print_spec = "+", "mid", 4

    def __new__(cls, arg0, arg1):
        if isinstance(arg0, Zero):
            return Broadcast(arg1, arg0.dim)
        elif isinstance(arg1, Zero):
            return Broadcast(arg0, arg1.dim)
        elif arg0 == arg1:
            from keops.python_engine.formulas.variables.IntCst import IntCst
            return IntCst(2) * arg0
        else:
            return super(Add, cls).__new__(cls)

    def ScalarOp(self, out, arg0, arg1):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return f"{out.id} = {arg0.id}+{arg1.id};\n"

    def DiffT(self, v, gradin):
        fa, fb = self.children
        return fa.Grad(v, gradin) + fb.Grad(v, gradin)
