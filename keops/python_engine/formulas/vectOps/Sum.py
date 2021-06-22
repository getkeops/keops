from keops.python_engine.utils.code_gen_utils import c_zero_float, VectApply
from keops.python_engine.formulas.Operation import Operation
from keops.python_engine.formulas.variables.Zero import Zero


##########################
######    Sum        #####
##########################

class Sum_Impl(Operation):
    # the summation operation
    string_id = "Sum"
    dim = 1

    def Op(self, out, table, arg):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return out.assign(c_zero_float) + VectApply(self.ScalarOp, out, arg)

    def ScalarOp(self, out, arg):
        return f"{out.id} += {arg.id};\n"

    def DiffT(self, v, gradin):
        from keops.python_engine.formulas.vectOps.SumT import SumT
        f = self.children[0]
        return f.DiffT(v, SumT(gradin, f.dim))
        

# N.B. The following separate function should theoretically be implemented
# as a __new__ method of the previous class, but this can generate infinite recursion problems
def Sum(arg):
    if isinstance(arg, Zero):
        return Zero(1)
    else:
        return Sum_Impl(arg)
