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

    def is_linear(self, v):
        return self == v

    def __init__(self, ind=None, dim=None, cat=None, params=None, label=None):
        # N.B. init via params keyword is used for compatibility with base class.
        if ind is None:
            # here we assume dim and cat are also None, and
            # that params is a tuple containing ind, dim, cat
            ind, dim, cat = params
        super().__init__(params=(ind, dim, cat))
        if label is None:
            # N.B. label is just a string used as an alias when printing the formulas ; it plays no role in computations.
            label = chr(ord("a") + ind) if ind >= 0 else chr(944 - ind)
        self.ind = ind
        self.dim = dim
        self.cat = cat
        self.label = label
        self.Vars_ = {self}

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
