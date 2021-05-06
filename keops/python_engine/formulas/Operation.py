from keops.python_engine.code_gen_utils import new_c_varname, c_array, VectApply, c_variable, cast_to, \
    VectCopy
from keops.python_engine.tree_class import tree

###################
## Base class
###################

class Operation(tree):

    # Base class for all keops building block operations in a formula

    def __init__(self, *args):
        # *args are other instances of Operation, they are the child operations of self
        self.children = args
        self.params = ()
        # The variables in the current formula is the union of the variables in the child operations.
        # Note that this requires implementing properly __eq__ and __hash__ methods in Var class
        self.Vars_ = set.union(*(arg.Vars_ for arg in args)) if len(args)>0 else set()

    def Vars(self, cat="all"):
        # if cat=="all", returns the list of all variables in a formula, stored in self.Vars_
        # if cat is an integer between 0 and 2, returns the list of variables v such that v.cat=cat
        if cat=="all":
            return list(self.Vars_)
        else:
            res = []
            for v in self.Vars_:
                if v.cat == cat:
                    res.append(v)
            return res

    def __call__(self, out, table):
        # returns the C++ code string corresponding to the evaluation of the formula
        # - out is a c_variable in which the result of the evaluation is stored
        # - table is the list of c_variables corresponding to actual local variables
        # required for evaluation : each Var(ind,*,*) corresponds to table[ind]
        string = f"\n{{\n// Starting code block for {self.__repr__()}.\n\n"
        args = []
        # Evaluation of the child operations
        for child in self.children:
            if isinstance(child,Var):
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
        string += f"\n\n// Finished code block for {self.__repr__()}.\n}}\n\n"
        return string

    def Grad(self, v, gradin):
        if gradin.dim != self.dim:
            raise ValueError("incompatible dimensions")
        return self.DiffT(v, gradin)

    def __mul__(self, other):
        """f*g redirects to Mult(f,g)"""
        from keops.python_engine.formulas.basicMathOps.Mult import Mult
        return Mult(self, other)

    def __rmul__(self, other):
        """g*f redirects to Mult(f,g)"""
        from keops.python_engine.formulas.basicMathOps.Mult import Mult
        return Mult(self, other)

    def __truediv__(self, other):
        """f/g redirects to Divide(f,g)"""
        from keops.python_engine.formulas.basicMathOps.Divide import Divide
        return Divide(self, other)

    def __add__(self, other):
        """f+g redirects to Add(f,g)"""
        from keops.python_engine.formulas.basicMathOps.Add import Add
        return Add(self, other)

    def __sub__(self, other):
        """f-g redirects to Subtract(f,g)"""
        from keops.python_engine.formulas.basicMathOps.Subtract import Subtract
        return Subtract(self, other)

    def __neg__(self):
        """-f redirects to Minus(f)"""
        from keops.python_engine.formulas.basicMathOps.Minus import Minus
        return Minus(self)

    def __pow__(self, other):
        """f**2 redirects to Square(f)"""
        from keops.python_engine.formulas.basicMathOps.Square import Square
        if other==2:
            return Square(self)
        else:
            raise ValueError("not implemented")

    def __or__(self, other):
        """f|g redirects to Scalprod(f,g)"""
        from keops.python_engine.formulas.basicMathOps.Scalprod import Scalprod
        return Scalprod(self, other)


class VectorizedScalarOp(Operation):
    # class for operations that are vectorized or broadcasted
    # scalar operations,
    # such as Exp(f), Cos(f), Mult(f,g), Subtract(f,g), etc.

    def __init__(self, *args):
        dims = set(arg.dim for arg in args)
        if len(dims)>2 or (len(dims)==2 and min(dims)!=1):
            raise ValueError("dimensions are not compatible for VectorizedScalarOp")
        super().__init__(*args)

    @property
    def dim(self):
        # dim gives the output dimension of the operation,
        # here it is the same as the output dimension of the child operation
        return max(child.dim for child in self.children)

    def Op(self, out, table, *arg):
        # Atomic evaluation of the operation : it consists in a simple
        # for loop around the call to the correponding scalar operation
        return VectApply(self.ScalarOp, out, *arg)



#######################
## Var operation
#######################

class Var(Operation):
    # Var operation class. Var(ind,dim,cat) is a symbolic
    # object that encodes an input tensor in the call to the
    # KeOps routine, where
    # - ind gives the position of the input tensor in the list of tensors sent to the routine
    # - dim gives the "dimension" of the data : each input tensor is interpreted as a matrix
    # of size (n,dim), where n is dynamically handled and dim is known at compile time.
    # - cat is the "category" of the variable : either a "i"-indexed variable (cat=0),
    # a "j"-indexed variable (cat=1), or a parameter variable (cat=2)

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
        return type(self)==type(other) and self.ind == other.ind and self.dim == other.dim and self.cat == other.cat
    def __hash__(self):
        return hash((self.ind,self.dim,self.cat))

    def Op(self, out, table):
        return VectCopy(out, table[self.ind], cast=False)

    # Assuming that the gradient wrt. Var is GRADIN, how does it affect V ?
    # Var::DiffT<V, grad_input> = grad_input   if V == Var (in the sense that it represents the same symb. var.)
    #                             Zero(V::DIM) otherwise
    def DiffT(self,v,gradin):
        return gradin if v==self else Zero(v.dim)


###################
## Constants
###################


class Zero(Operation):
    # zero operation : encodes a vector of zeros
    string_id = "Zero"

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.params = (dim,)

    # custom __eq__ method
    def __eq__(self, other):
        return type(self) == type(other) and self.dim == other.dim

    def Op(self, out, table):
        zero = c_variable("float", "0.0f")
        return out.assign(zero)

    def DiffT(self, v, gradin):
        return Zero(v.dim)


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
        return Zero(v.dim)



##########################
#####    Broadcast    ####
##########################

# N.B. this is used internally
def Broadcast(arg, dim):
    if arg.dim == dim or dim == 1:
        return arg
    elif arg.dim == 1:
        return SumT(arg, dim)
    else:
        raise ValueError("dimensions are not compatible for Broadcast operation")


