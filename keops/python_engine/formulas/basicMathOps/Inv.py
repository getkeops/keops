from keops.python_engine.formulas.VectorizedScalarOp import VectorizedScalarOp


##########################
######    INVERSE : Inv<F>        #####
##########################

class Inv(VectorizedScalarOp):
    """the "Inv" vectorized operation"""
    string_id = "Inv"
    print_spec = "1/", "pre", 2

    def ScalarOp(self, out, arg):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return f"{out.id} = {keops_rcp(arg)};\n"

    def DiffT(self, v, gradin):
        f = self.children[0]
        DiffTF = f.DiffT(v, gradin)
        return -DiffTF(v, Mult(Inv(f)**2)*gradin)

