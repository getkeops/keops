from keopscore.utils.code_gen_utils import VectCopy
from keopscore.formulas.Operation import Operation


#######################
## Var operation
#######################


class Var(Operation):
    """Var operation class. Var(ind,dim,cat) is a symbolic
    object that encodes an input tensor in the call to the
    KeOps routine, where
    - ind gives the position of the input tensor in the list of tensors sent to the routine
    - dim gives the "dimension" of the data : each input tensor is interpreted as a matrix
    of size (n,dim), where n is dynamically handled and dim is known at compile time.
    - cat is the "category" of the variable : either a "i"-indexed variable (cat=0),
    a "j"-indexed variable (cat=1), or a parameter variable (cat=2)"""

    string_id = "Var"

    def __init__(self, ind, dim, cat):
        super().__init__()
        self.ind = ind
        self.dim = dim
        self.cat = cat
        self.Vars_ = {self}
        self.params = (ind, dim, cat)

    # custom __eq__ and __hash__ methods, required to handle properly the union of two sets of Var objects
    def __eq__(self, other):
        return (
            type(self) == type(other)
            and self.ind == other.ind
            and self.dim == other.dim
            and self.cat == other.cat
        )

    def __hash__(self):
        return hash((self.ind, self.dim, self.cat))

    def Op(self, out, table):
        return VectCopy(out, table[self.ind])

    # Assuming that the gradient wrt. Var is GRADIN, how does it affect V ?
    # Var::DiffT<V, grad_input> = grad_input   if V == Var (in the sense that it represents the same symb. var.)
    #                             Zero(V::DIM) otherwise
    def DiffT(self, v, gradin):
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
        return self.Vars(cat)

    def notchunked_vars(self, cat):
        return set()
