from keops.python_engine.formulas.Operation import Operation


#/////////////////////////////////////////////////////////////////////////
#////      ComplexExp1j                           ////
#/////////////////////////////////////////////////////////////////////////


class ComplexExp1j(Operation):
    
    string_id = "ComplexExp1j"

    def __init__(self, f):
        self.dim = 2*f.dim
        super().__init__(f)
    
    def Op(self, out, table, inF):
        from keops.python_engine.utils.math_functions import keops_sincos
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        forloop, i = c_for_loop(0, self.dim, 2, pragma_unroll=True)
        body = keops_sincos(inF[i/2], out[i]+1, out[i])
        return forloop(body)
                
    def DiffT(self, v, gradin):
        from keops.python_engine.formulas.complex.Real2Complex import Real2Complex
        from keops.python_engine.formulas.complex.Imag2Complex import Imag2Complex
        from keops.python_engine.formulas.basicMathsOps.Add import Add
        from keops.python_engine.formulas.maths.Cos import Cos
        from keops.python_engine.formulas.maths.Sin import Sin
        # building equivalent formula for autodiff
        f = self.children[0]
        AltFormula = Add(Real2Complex(Cos(f)) , Imag2Complex(Sin(f)))
        return AltFormula.DiffT(v, gradin)
