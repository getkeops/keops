from keopscore.utils.meta_toolbox.c_instruction import (
    c_comment,
    c_instruction,
    c_empty_instruction,
    c_instruction_from_string,
)
from keopscore.utils.meta_toolbox.c_block import c_block
from keopscore.utils.code_gen_utils import new_c_name, c_array, cast_to, c_variable
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

    def get_out_var(self, dtype):
        template_string_id = "out_" + self.string_id.lower()
        name = new_c_name(template_string_id)
        if self.dim == 1:
            return c_variable(dtype, name)
        else:
            return c_array(dtype, self.dim, name)

    def get_code_and_expr(self, dtype, table, i, j, tagI):
        out = self.get_out_var(dtype)
        code = out.declare() + self(out, table, i, j, tagI)
        return code, out

    def __call__(self, out, table, i, j, tagI):
        """returns the C++ code string corresponding to the evaluation of the formula
         - out is a c_variable in which the result of the evaluation is stored
         - table is the list of c_variables corresponding to actual local variables
        required for evaluation : each Var(ind,*,*) corresponds to table[ind]"""
        from keopscore.formulas.variables.Var import Var
        from keopscore.formulas.variables.IJ import I, J

        code = c_comment(f"Starting code block for {self.__repr__()}")
        if keopscore.debug_ops:
            print(f"Building code block for {self.__repr__()}")
            print("out=", out)
            print("dim of out : ", out.dim)
            print("table=", table)
            for v in table:
                print(f"dim of {v} : ", v.dim)
        if keopscore.debug_ops_at_exec:
            code += c_instruction(
                f'printf("\\n\\nComputing {self.__repr__()} :\\n")', set(), set()
            )
        # Evaluation of the child operations
        if len(self.children) > 0:
            code_args, args = zip(
                *(
                    child.get_code_and_expr(out.dtype, table, i, j, tagI)
                    for child in self.children
                )
            )
            code += sum(code_args, c_empty_instruction)
        else:
            args = ()
        # Finally, evaluation of the operation itself
        code += self.Op(out, table, *args)

        # some debugging helper :
        if keopscore.debug_ops_at_exec:
            for arg in args:
                code += c_instruction_from_string(arg.c_print)
            code += c_instruction_from_string(out.c_print)
            code += c_instruction_from_string(f'printf("\\n\\n");\n')
        if keopscore.debug_ops:
            print(f"Finished building code block for {self.__repr__()}")

        code += c_comment(f"Finished code block for {self.__repr__()}")
        code = c_block(code)
        return code

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
        return type(self)(*args, *self.params)

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
