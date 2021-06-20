
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

    def DiffT(self, v, gradin):
        # [ \partial_V Atan2(F, G) ] . gradin = [ (G / F^2 + G^2) . \partial_V F ] . gradin  - [ (F / F^2 + G^2) . \partial_V G ] . gradin
        f, g = self.children
        sqnormfg = f**2+g**2
        return f.DiffT(v, g*gradin/sqnormfg) - g.DiffT(v, f*gradin/sqnormfg)
