from keopscore.formulas.Operation import Operation
from keopscore.utils.meta_toolbox import c_zero_float, c_for_loop
from keopscore.utils.misc_utils import KeOps_Error

######################################################
######    ELEMENT "INJECTION" : IndexT(f,g,n)   ######
######################################################

# IndexT(f,g,n) returns the vector of size n with all values equal to 0, except the g-th value which is equal to f.
# It is identical to ElemT operation except that here the indexing value g is a formula instead of an integer constant.
# (also ordering of arguments differs.)


def IndexT(f, g, n):
    return IndexT_Factory(n)(f, g)


class IndexT_Impl(Operation):
    pass


class IndexT_Factory:

    def __init__(self, n):

        class Class(IndexT_Impl):

            string_id = "IndexT"
            linearity_type = "first"
            print_fun = lambda x, y: f"IndexT({x},{y},{n})"

            def __init__(self, f, g):
                super().__init__(f, g)
                if f.dim != 1 or g.dim != 1:
                    KeOps_Error("Inputs of IndexT should be scalar")
                self.dim = n
                self.n = n

            def Op(self, out, table, arga, argb):
                n = self.n
                loop1, k = c_for_loop(0, n, 1, pragma_unroll=True)
                res = loop1(out[k].assign(c_zero_float))
                res += out[argb.value].assign(arga.value)
                return res

            def DiffT(self, v, gradin):
                from keopscore.formulas.maths.Index import Index

                f, g = self.children
                return f.DiffT(v, Index(gradin, g))

            # parameters for testing the operation (optional)
            enable_test = False  # enable testing for this operation
            # N.B. test here can probably be adapted from ElemT one.

        self.Class = Class

    def __call__(self, f, g):
        return self.Class(f, g)
