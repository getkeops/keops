from keopscore.utils.code_gen_utils import c_variable, new_c_varname, c_array
from keopscore.utils.Tree import Tree
import keopscore
from keopscore.utils.misc_utils import KeOps_Error

###################
## Base class
###################


class Operation(Tree):
    """Base class for all keops building block operations in a formula"""

    linearity_type = None

    def is_linear(self, v):
        if self.linearity_type == "all":
            return all(f.is_linear(v) for f in self.children)
        elif self.linearity_type == "one":
            return sum(f.is_linear(v) for f in self.children) == 1
        elif self.linearity_type == "first":
            f = self.children[0]
            return f.is_linear(v)
        else:
            return False

    def __init__(self, *args, params=()):
        # *args are other instances of Operation, they are the child operations of self
        self.children = args
        self.params = params

        # The variables in the current formula is the union of the variables in the child operations.
        # Note that this requires implementing properly __eq__ and __hash__ methods in Var class.
        # N.B. We need to sort according to ind.
        set_vars = (
            set.union(*(set(arg.Vars_) for arg in args)) if len(args) > 0 else set()
        )
        self.Vars_ = sorted(list(set_vars), key=lambda v: v.ind)
    
    def renew(self, args=None, params=None, allow_fuse=True):
        if args is None:
            args = self.children
        if params is None:
            params = self.params
        return type(self)(*args, *params, allow_fuse=allow_fuse)

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

    def replace(self, old, new, cnt=[0]):
        # replace all occurences of subformula old by new in self.
        if self == old:
            cnt[0] += 1
            return new
        else:
            new_children = [child.replace(old, new, cnt) for child in self.children]
            return type(self)(*new_children, *self.params)

    def replace_and_count(self, old, new):
        cnt = [0]
        formula = self.replace(old, new, cnt)
        return formula, cnt[0]

    def __call__(self, out, table):
        """returns the C++ code string corresponding to the evaluation of the formula
         - out is a c_variable in which the result of the evaluation is stored
         - table is the list of c_variables corresponding to actual local variables
        required for evaluation : each Var(ind,*,*) corresponds to table[ind]"""
        from keopscore.formulas.variables.Var import Var

        string = f"\n{{\n// Starting code block for {self.__repr__()}.\n\n"
        if keopscore.debug_ops:
            print(f"Building code block for {self.__repr__()}")
            print("out=", out)
            print("dim of out : ", out.dim)
            print("table=", table)
            for v in table:
                print(f"dim of {v} : ", v.dim)
        if keopscore.debug_ops_at_exec:
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
        if keopscore.debug_ops_at_exec:
            for arg in args:
                string += arg.c_print
            string += out.c_print
            string += f'printf("\\n\\n");\n'
        if keopscore.debug_ops:
            print(f"Finished building code block for {self.__repr__()}")

        string += f"\n\n// Finished code block for {self.__repr__()}.\n}}\n\n"
        return string

    def __mul__(self, other):
        """f*g redirects to Mult(f,g)"""
        from keopscore.formulas.maths.Mult import Mult

        return Mult(self, int2Op(other))

    def __rmul__(self, other):
        from keopscore.formulas.maths.Mult import Mult

        return Mult(int2Op(other), self)

    def __truediv__(self, other):
        """f/g redirects to Divide(f,g)"""
        from keopscore.formulas.maths.Divide import Divide

        return Divide(self, int2Op(other))

    def __rtruediv__(self, other):
        if other == 1:
            from keopscore.formulas.maths.Inv import Inv

            return Inv(self)
        else:
            return int2Op(other) / self

    def __add__(self, other):
        """f+g redirects to Add(f,g)"""
        from keopscore.formulas.maths.Add import Add

        return Add(self, int2Op(other))

    def __radd__(self, other):
        """f+g redirects to Add(f,g)"""
        return int2Op(other) + self

    def __sub__(self, other):
        """f-g redirects to Subtract(f,g)"""
        from keopscore.formulas.maths.Subtract import Subtract

        return Subtract(self, int2Op(other))

    def __rsub__(self, other):
        """f-g redirects to Subtract(f,g)"""
        return int2Op(other) - self

    def __neg__(self):
        """-f redirects to Minus(f)"""
        from keopscore.formulas.maths.Minus import Minus

        return Minus(self)

    def __pow__(self, other):
        if other == 2:
            """f**2 redirects to Square(f)"""
            from keopscore.formulas.maths.Square import Square

            return Square(self)
        elif isinstance(other, int):
            """f**m with m integer redirects to Pow(f,m)"""
            from keopscore.formulas.maths.Pow import Pow

            return Pow(self, other)
        else:
            from keopscore.formulas.maths.Powf import Powf

            raise Powf(self, other)

    def __or__(self, other):
        """f|g redirects to Scalprod(f,g)"""
        from keopscore.formulas.maths.Scalprod import Scalprod

        return Scalprod(self, other)

    def __eq__(self, other):
        return (
            type(self) == type(other)
            and self.children == other.children
            and self.params == other.params
        )

    def __lt__(self, other):
        """f<g redirects to LessThan(f,g)"""
        from keopscore.formulas.maths.LessThan import LessThan

        return LessThan(self, int2Op(other))

    def __gt__(self, other):
        """f>g redirects to LessThan(g,f)"""
        return int2Op(other) < self

    def __le__(self, other):
        """f<=g redirects to LessOrEqual(f,g)"""
        from keopscore.formulas.maths.LessOrEqual import LessOrEqual

        return LessOrEqual(self, int2Op(other))

    def __ge__(self, other):
        """f>=g redirects to LessOrEqual(g,f)"""
        return int2Op(other) <= self

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
        return self.renew(args=args)

    enable_test = False
    disable_testgrad = False


def int2Op(x):
    if isinstance(x, int):
        from keopscore.formulas.variables.IntCst import IntCst

        return IntCst(x)
    elif isinstance(x, Operation):
        return x
    else:
        KeOps_Error("invalid type : " + str(type(x)))


##########################
#####    Broadcast    ####
##########################


# N.B. this is used internally
def Broadcast(arg, dim):
    from keopscore.formulas.maths import SumT

    if arg.dim == dim or dim == 1:
        return arg
    elif arg.dim == 1:
        return SumT(arg, dim)
    else:
        KeOps_Error("dimensions are not compatible for Broadcast operation")





class FusedOp(Operation):

    def recursive_str(self):
        return self.parent_op.recursive_str()
    
    def __init__(self, *args, params=()):
        parent_op, ind_child_op = params
        child_op = parent_op.children[ind_child_op]
        super().__init__(*args, params=params)
        self.parent_op, self.ind_child_op, self.child_op = (
            parent_op,
            ind_child_op,
            child_op,
        )
        self.linearity_type = parent_op.linearity_type
        self.is_linear = parent_op.is_linear
    
    def renew(self, args=None, params=None, allow_fuse=True):
        i, m = self.ind_child_op, len(self.child_op.children)
        args_child = args[i : i + m]
        new_child = self.child_op.renew(args=args_child, allow_fuse=allow_fuse)
        args_parent = args[:i] + [new_child] + args[i + m :]
        return self.parent_op.renew(args=args_parent, params=params, allow_fuse=allow_fuse)

    def ScalarOp(self, out, *args):
        i, m = self.ind_child_op, len(self.child_op.children)
        args_child = args[i : i + m]
        out_child = c_variable(out.dtype)
        str_child = self.child_op.ScalarOp(out_child, *args_child)
        args_parent = args[:i] + (out_child,) + args[i + m :]
        str_parent = self.parent_op.ScalarOp(out, *args_parent)
        return "{" + out_child.declare() + str_child + str_parent + "}"

    def DiffT(self, v, gradin):
        return self.parent_op.DiffT(v, gradin)