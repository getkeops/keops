from typing import Any
from keopscore.utils.meta_toolbox import c_empty_instruction
from keopscore.formulas.Operation import Operation
from keopscore.utils.unique_object import unique_object

#######################
## Var operation
#######################


class Var_Impl(Operation):
    pass


class Var_Factory(metaclass=unique_object):

    def __init__(self, ind, dim, cat):

        # N.B. label is just a string used as an alias when printing the formulas ; it plays no role in computations.
        label = chr(ord("a") + ind) if ind >= 0 else chr(944 - ind)

        class Class(Var_Impl):
            """Var operation class. Var(ind,dim,cat) is a symbolic
            object that encodes an input tensor in the call to the
            KeOps routine, where
            - ind gives the position of the input tensor in the list of tensors sent to the routine
            - dim gives the "dimension" of the data : each input tensor is interpreted as a matrix
            of size (n,dim), where n is dynamically handled and dim is known at compile time.
            - cat is the "category" of the variable : either a "i"-indexed variable (cat=0),
            a "j"-indexed variable (cat=1), or a parameter variable (cat=2)"""

            string_id = f"Var({ind},{dim},{cat})"
            print_fun = lambda: Class.string_id

            def is_linear(self, v):
                return self == v

            def __init__(self):
                super().__init__()
                self.ind = ind
                self.dim = dim
                self.cat = cat
                self.set_label(label)
                self.Vars_ = {self}

            def set_label(self, label):
                if label is None:
                    # N.B. label is just a string used as an alias when printing the formulas ; it plays no role in computations.
                    self.label = chr(ord("a") + ind) if ind >= 0 else chr(944 - ind)
                else:
                    self.label = label
            
            def set_vars(self):
                pass

            def __hash__(self):
                return hash((self.ind, self.dim, self.cat))

            def get_code_and_expr(self, dtype, table, i, j, tagI):
                return c_empty_instruction, table[self.ind]

            def Op(self, out, table):
                return out.copy(table[self.ind])

            # Assuming that the gradient wrt. Var is GRADIN, how does it affect V ?
            # Var::DiffT<V, grad_input> = grad_input   if V == Var (in the sense that it represents the same symb. var.)
            #                             Zero(V::DIM) otherwise
            def DiffT_fun(self, v, gradin):
                from keopscore.formulas.variables.Zero import Zero

                return gradin if v == self else Zero(v.dim)

            def chunked_version(self, dimchk):
                return Var(self.ind, dimchk, self.cat)

            @property
            def is_chunkable(self):
                return True

            def chunked_formulas(self, dimchk):
                return []

            @property
            def num_chunked_formulas(self):
                return 0

            def post_chunk_formula(self, ind):
                return Var(self.ind, self.dim, self.cat)

            def chunked_vars(self, cat):
                return self.Vars(self.cat)

            def notchunked_vars(self, cat):
                return set()

        self.Class = Class

    def __call__(self):
        return self.Class()


def Var(ind, dim, cat, label=None):
    res = Var_Factory(ind, dim, cat)()
    res.set_label(label)
    return res
