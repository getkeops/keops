from keops.python_engine.formulas.VectorizedScalarOp import VectorizedScalarOp

##########################
######    Sign        #####
##########################

class Sign(VectorizedScalarOp):
    """the sign vectorized operation"""
    string_id = "Sign"

    def ScalarOp(self, out, arg):
        from keops.python_engine.utils.math_functions import keops_sign
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return out.assign(keops_sign(arg))
      
    def DiffT(self, v, gradin):
        from keops.python_engine.formulas.variables.Zero import Zero
        return Zero(v.dim)
