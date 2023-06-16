import keopscore
from keopscore.utils.code_gen_utils import new_c_varname, c_array
from keopscore.utils.Tree import Tree
from keopscore import debug_ops, debug_ops_at_exec
from keopscore.utils.misc_utils import KeOps_Error

###################
## Base class
###################


class Operation(Tree):
    """Base class for all keops building block operations in a formula"""

    def __init__(self, *args, shapes=None, params=()):
        super().__init__(*args, shapes=shapes, params=params)

        self.params = params
        self.shapes = shapes
        self.check_shapes()

        # The variables in the current formula is the union of the variables in the child operations.
        # Note that this requires implementing properly __eq__ and __hash__ methods in Var class.
        # N.B. We need to sort according to ind.
        set_vars = (
            set.union(*(set(arg.Vars_) for arg in args)) if len(args) > 0 else set()
        )
        self.Vars_ = sorted(list(set_vars), key=lambda v: v.ind)

    def check_shapes(self):
        # used for checking input shapes or dims of args, depending on some rules.
        # We do nothing by default ; see VectorizedScalarOp for a derived implementation
        pass

    def Vars(self, cat="all"):
        # if cat=="all", returns the list of all variables in a formula, stored in self.Vars_
        # if cat is an integer between 0 and 2, returns the list of variables v such that v.cat=cat
        if cat == "all":
            return self.Vars_
        else:
            res = []
            for v in self.Vars_:
                if v.cat == cat:
                    res.append(v)
            return res

    def replace(self, old, new):
        # replace all occurences of subformula old by new in self.
        if self == old:
            return new
        else:
            new_children = [child.replace(old, new) for child in self.children]
            return type(self)(*new_children, *self.params)

    def __call__(self, out, table):
        """returns the C++ code string corresponding to the evaluation of the formula
         - out is a c_variable in which the result of the evaluation is stored
         - table is the list of c_variables corresponding to actual local variables
        required for evaluation : each Var(ind,*,*) corresponds to table[ind]"""
        from keopscore.formulas.variables.Var import Var

        string = f"\n{{\n// Starting code block for {self.__repr__()}.\n\n"
        if debug_ops:
            print(f"Building code block for {self.__repr__()}")
            print("out=", out)
            print("dim of out : ", out.dim)
            print("table=", table)
            for v in table:
                print(f"dim of {v} : ", v.dim)
        if debug_ops_at_exec:
            string += f'printf("\\n\\nComputing {self.__repr__()} :\\n");\n'
        args = []
        # Evaluation of the child operations
        for child in self.children:
            if isinstance(child, Var):
                # if the child of the operation is a Var, we do not need to evaluate it,
                # we simply record the corresponding c_variable
                arg = table[child.ind]
            else:
                # otherwise, we need to evaluate the child operation.
                # We first create a new c_array to store the result of the child operation.
                # This c_array must have a unique name in the code, to avoid conflicts
                # when we will recursively evaluate nested operations.
                template_string_id = "out_" + child.string_id.lower()
                arg_name = new_c_varname(template_string_id)
                arg = c_array(out.dtype, child.dim, arg_name)
                # Now we append into string the C++ code to declare the array
                string += f"{arg.declare()}\n"
                # Now we evaluate the child operation and append the result into string
                string += child(arg, table)
            args.append(arg)
        # Finally, evaluation of the operation itself
        string += self.Op(out, table, *args)

        # some debugging helper :
        if debug_ops_at_exec:
            for arg in args:
                string += arg.c_print
            string += out.c_print
            string += f'printf("\\n\\n");\n'
        if debug_ops:
            print(f"Finished building code block for {self.__repr__()}")

        string += f"\n\n// Finished code block for {self.__repr__()}.\n}}\n\n"
        return string

    def __mul__(self, other):
        """f*g redirects to Mult(f,g)"""
        return keopscore.formulas.maths.Mult(self, int2Op(other))

    def __rmul__(self, other):
        return keopscore.formulas.maths.Mult(int2Op(other), self)

    def __truediv__(self, other):
        """f/g redirects to Divide(f,g)"""
        return keopscore.formulas.maths.Divide(self, int2Op(other))

    def __rtruediv__(self, other):
        if other == 1:
            return keopscore.formulas.maths.Inv(self)
        else:
            return int2Op(other) / self

    def __add__(self, other):
        """f+g redirects to Add(f,g)"""
        return keopscore.formulas.maths.Add(self, int2Op(other))

    def __radd__(self, other):
        """f+g redirects to Add(f,g)"""
        return int2Op(other) + self

    def __sub__(self, other):
        """f-g redirects to Subtract(f,g)"""
        return keopscore.formulas.maths.Subtract(self, int2Op(other))

    def __rsub__(self, other):
        """f-g redirects to Subtract(f,g)"""
        return int2Op(other) - self

    def __neg__(self):
        """-f redirects to Minus(f)"""
        return keopscore.formulas.maths.Minus(self)

    def __pow__(self, other):
        if other == 2:
            """f**2 redirects to Square(f)"""
            return keopscore.formulas.maths.Square(self)
        elif isinstance(other, int):
            """f**m with m integer redirects to Pow(f,m)"""
            return keopscore.formulas.maths.Pow(self, other)
        else:
            return keopscore.formulas.maths.Powf(self, other)

    def __or__(self, other):
        """f|g redirects to Scalprod(f,g)"""
        return keopscore.formulas.maths.Scalprod(self, other)

    def Op(self, out, table, param):
        pass

    def chunked_version(self, dimchk):
        return None

    @property
    def is_chunkable(self):
        return False

    def chunked_formulas(self, dimchk):
        res = []
        for child in self.children:
            res += child.chunked_formulas(dimchk)
        return res

    @property
    def num_chunked_formulas(self):
        return sum([child.num_chunked_formulas for child in self.children])

    def post_chunk_formula(self, ind):
        args = []
        for child in self.children:
            args.append(child.post_chunk_formula(ind))
            ind += child.num_chunked_formulas
        return type(self)(*args, *self.params)

    enable_test = False
    disable_testgrad = False


def int2Op(x):
    if isinstance(x, int):
        return keopscore.formulas.variables.IntCst(x)
    elif isinstance(x, Operation):
        return x
    else:
        KeOps_Error("invalid type : " + str(type(x)))


##########################
#####    Broadcast    ####
##########################


# helpers for broadcasted operators
# N.B. these are used internally only


def Broadcast(arg, dim):
    if arg.dim == dim or dim == 1:
        return arg
    elif arg.dim == 1:
        return keopscore.formulas.maths.SumT(arg, dim)
    else:
        KeOps_Error("dimensions are not compatible for Broadcast operation")


def BroadcastT(arg, dim):
    if arg.dim == dim:
        return arg
    elif dim == 1:
        return keopscore.formulas.maths.Sum(arg)
    else:
        KeOps_Error("dimensions are not compatible for BroadcastT operation")
