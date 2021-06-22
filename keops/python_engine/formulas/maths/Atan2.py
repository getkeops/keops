
from keops.python_engine.formulas.VectorizedScalarOp import VectorizedScalarOp

#//////////////////////////////////////////////////////////////
#////                 ATAN2 :  Atan2< F, G >               ////
#//////////////////////////////////////////////////////////////

class Atan2(VectorizedScalarOp):

    string_id = "Atan2"

    def ScalarOp(self, out, arg0, arg1):
        from keops.python_engine.utils.math_functions import keops_atan2
        # returns the atomic piece of c++ code to evaluate the function on arg0 and arg1 and return
        # the result in out
        return out.assign(keops_atan2(arg0, arg1))
    
    @property
    def Derivative(self):  
        # this is buggy, must investigate...
        raise ValueError("not implemented")
        f, g = self.children
        return g/(f**2+g**2), -f/(f**2+g**2)
    
