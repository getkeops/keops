from keopscore.formulas.Operation import Operation
from keopscore.utils.meta_toolbox.c_lvalue import c_value
from keopscore.utils.code_gen_utils import c_zero_float, c_for_loop
from keopscore.utils.misc_utils import KeOps_Error

######################################################
######    ELEMENT "INJECTION" : IndexT(f,g,n)   ######
######################################################

# IndexT(f,g,n) returns the vector of size n with all values equal to 0, except the g-th value which is equal to f.
# It is identical to ElemT operation except that here the indexing value g is a formula instead of an integer constant.
# (also ordering of arguments differs.)


class IndexT(Operation):
    string_id = "IndexT"
    linearity_type = "first"

    def __init__(self, f, g, n=None, params=None):
        # N.B. init via params keyword is used for compatibility with base class.
        if n is None:
            # here we assume params is a tuple containing n
            (n,) = params
        super().__init__(f, g, params=(n,))
        if f.dim != 1 or g.dim != 1:
            KeOps_Error("Inputs of IndexT should be scalar")
        self.dim = n
        self.n = n

    def Op(self, out, table, arga, argb):
        n = self.n
        loop1, k = c_for_loop(0, n, 1, pragma_unroll=True)
        string = loop1(out[k].assign(c_zero_float))
        string += out[c_value(argb)].assign(c_value(arga))
        return string

    def DiffT(self, v, gradin):
        from keopscore.formulas.maths.Index import Index

        f, g = self.children
        return f.DiffT(v, Index(gradin, g))

    # parameters for testing the operation (optional)
    enable_test = False  # enable testing for this operation
    # N.B. test here can probably be adapted from ElemT one.
