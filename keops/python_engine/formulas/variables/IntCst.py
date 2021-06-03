from keops.python_engine.utils.code_gen_utils import cast_to
from keops.python_engine.formulas.maths.Operation import Operation


class IntCst(Operation):
    # constant integer "operation"
    string_id = "IntCst"
    print_spec = "", "pre", 0

    def __init__(self, val):
        super().__init__()
        self.val = val
        self.dim = 1
        self.params = (val,)

    # custom __eq__ method
    def __eq__(self, other):
        return type(self) == type(other) and self.val == other.val

    def Op(self, out, table):
        return f"*{out.id} = {cast_to(out.dtype)}((float){self.val});\n"

    def DiffT(self, v, gradin):
        from keops.python_engine.formulas import Zero
        return Zero(v.dim)