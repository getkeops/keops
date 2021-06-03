from keops.python_engine.utils.code_gen_utils import c_zero_float, VectApply
from keops.python_engine.formulas.maths.Operation import Operation
from keops.python_engine.formulas.variables.Zero import Zero


##########################
######    Sum        #####
##########################

class Sum(Operation):
    # the summation operation
    string_id = "Sum"
    dim = 1

    def __new__(cls, arg):
        if isinstance(arg, Zero):
            return Zero(1)
        else:
            return super(Sum, cls).__new__(cls)

    def Op(self, out, table, arg):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return out.assign(c_zero_float) + VectApply(self.ScalarOp, out, arg)

    def ScalarOp(self, out, arg):
        return f"{out.id} += {arg.id};\n"

    def DiffT(self, v, gradin):
        from keops.python_engine.formulas.maths.SumT import SumT
        f = self.children[0]
        return f.Grad(v, SumT(gradin, f.dim))