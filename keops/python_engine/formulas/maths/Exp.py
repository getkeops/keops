from keops.python_engine.formulas.maths.VectorizedScalarOp import VectorizedScalarOp


##########################
######    Exp        #####
##########################

class Exp(VectorizedScalarOp):
    """the exponential vectorized operation"""
    string_id = "Exp"
    def ScalarOp(self,out,arg):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return f"{out.id} = {keops_exp(arg)};\n"
    def DiffT(self,v,gradin):
        # [\partial_V exp(F)].gradin = exp(F) * [\partial_V F].gradin
        f = self.children[0]
        return f.Grad(v,Exp(f)*gradin)


def keops_exp(x):
    # returns the C++ code string for the exponential function applied to a C++ variable
    # - x must be of type c_variable
    if x.dtype in ["float","double"]:
        return f"exp({x.id})"
    else:
        raise ValueError("not implemented.")