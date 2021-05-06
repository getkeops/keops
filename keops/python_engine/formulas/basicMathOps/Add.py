from keops.python_engine.formulas.Operation import VectorizedScalarOp, Zero, Broadcast, IntCst

##########################
######    Add        #####
##########################

class Add_(VectorizedScalarOp):
    """the binary addition operation"""
    string_id = "Add"
    print_spec = "+", "mid", 4

    def ScalarOp(self, out, arg0, arg1):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return f"{out.id} = {arg0.id}+{arg1.id};\n"

    def DiffT(self, v, gradin):
        fa, fb = self.children
        return fa.Grad(v, gradin) + fb.Grad(v, gradin)


def Add(arg0, arg1):
    if isinstance(arg0, Zero):
        return Broadcast(arg1, arg0.dim)
    elif isinstance(arg1, Zero):
        return Broadcast(arg0, arg1.dim)
    elif arg0 == arg1:
        return IntCst(2) * arg0
    else:
        return Add_(arg0, arg1)