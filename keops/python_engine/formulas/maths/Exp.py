from keops.python_engine.formulas.VectorizedScalarOp import VectorizedScalarOp


##########################
######    Exp        #####
##########################



class Exp(VectorizedScalarOp):
    """the exponential vectorized operation"""
    string_id = "Exp"

    def ScalarOp(self, out, arg):
        from keops.python_engine.utils.math_functions import keops_exp
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return out.assign(keops_exp(arg))

    def DiffT(self, v, gradin):
        # [\partial_V exp(F)].gradin = exp(F) * [\partial_V F].gradin
        f = self.children[0]
        return f.Grad(v, Exp(f) * gradin)
