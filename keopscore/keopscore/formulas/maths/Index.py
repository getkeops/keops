from keopscore.formulas.Operation import Operation
from keopscore.utils.code_gen_utils import c_value
from keopscore.utils.misc_utils import KeOps_Error

##########################################################
######    INDEXING : Index(f,g) (aka get_item)    ########
##########################################################


class Index(Operation):
    string_id = "Index"
    print_spec = ("[", "]"), "item", 1
    linearity_type = "first"

    def __init__(self, f, g, m=None, params=()):
        # N.B. params keyword is used for compatibility with base class, but should always equal ()
        if params != ():
            KeOps_Error("There should be no parameter.")
        if g.dim != 1:
            KeOps_Error("dimension of index variable should be 1.")
        super().__init__(f, g)
        self.dim = 1

    def Op(self, out, table, arga, argb):
        return c_value(out).assign(arga[c_value(argb)])

    def DiffT(self, v, gradin):
        from keopscore.formulas.maths.IndexT import IndexT

        f, g = self.children
        return f.DiffT(v, IndexT(gradin, g, f.dim))

    # parameters for testing the operation (optional)
    enable_test = False  # enable testing for this operation
    # N.B. test here can probably be adapted from Elem one.
