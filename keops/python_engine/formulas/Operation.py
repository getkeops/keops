from keops.python_engine.utils.code_gen_utils import new_c_varname, c_array
from keops.python_engine.utils.Tree import Tree
from keops.python_engine import debug_ops


###################
## Base class
###################

class Operation(Tree):
    """Base class for all keops building block operations in a formula"""
    def __init__(self, *args, params=()):
        # *args are other instances of Operation, they are the child operations of self
        self.children = args
        self.params = params
        # The variables in the current formula is the union of the variables in the child operations.
        # Note that this requires implementing properly __eq__ and __hash__ methods in Var class
        self.Vars_ = set.union(*(arg.Vars_ for arg in args)) if len(args) > 0 else set()

    def Vars(self, cat="all"):
        # if cat=="all", returns the list of all variables in a formula, stored in self.Vars_
        # if cat is an integer between 0 and 2, returns the list of variables v such that v.cat=cat
        if cat == "all":
            return list(self.Vars_)
        else:
            res = []
            for v in self.Vars_:
                if v.cat == cat:
                    res.append(v)
            return res

    def __call__(self, out, table):
        """returns the C++ code string corresponding to the evaluation of the formula
          - out is a c_variable in which the result of the evaluation is stored
          - table is the list of c_variables corresponding to actual local variables
         required for evaluation : each Var(ind,*,*) corresponds to table[ind]"""
        from keops.python_engine.formulas.variables.Var import Var

        string = f"\n{{\n// Starting code block for {self.__repr__()}.\n\n"
        if debug_ops:
            string += f'std::cout << std::endl << std::endl << "Computing {self.__repr__()} :" << std::endl;\n'
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
        if debug_ops:
            for arg in args:
                string += arg.c_print()
            string += out.c_print()
            string += f'std::cout << std::endl << std::endl;\n'
            
        string += f"\n\n// Finished code block for {self.__repr__()}.\n}}\n\n"
        return string
        
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

    def __rtruediv__(self, other):
        if other==1:
            from keops.python_engine.formulas.basicMathOps.Inv import Inv
            return Inv(self)
        else:
            return int2Op(other) / self

    def __add__(self, other):
        """f+g redirects to Add(f,g)"""
        from keops.python_engine.formulas.basicMathOps.Add import Add
        return Add(self, other)

    def __radd__(self, other):
        """f+g redirects to Add(f,g)"""
        return int2Op(other) + self

    def __sub__(self, other):
        """f-g redirects to Subtract(f,g)"""
        from keops.python_engine.formulas.basicMathOps.Subtract import Subtract
        return Subtract(self, other)

    def __rsub__(self, other):
        """f-g redirects to Subtract(f,g)"""
        return int2Op(other) - self

    def __neg__(self):
        """-f redirects to Minus(f)"""
        from keops.python_engine.formulas.basicMathOps.Minus import Minus
        return Minus(self)

    def __pow__(self, other):
        """f**2 redirects to Square(f)"""
        from keops.python_engine.formulas.maths.Square import Square
        if other == 2:
            return Square(self)
        else:
            raise ValueError("not implemented")

    def __or__(self, other):
        """f|g redirects to Scalprod(f,g)"""
        from keops.python_engine.formulas.vectOps.Scalprod import Scalprod
        return Scalprod(self, other)
        
    def __eq__(self, other):
        return type(self) == type(other) and self.params == other.params

    def Op(self, out, table, param):
        pass



def int2Op(x):
    if isinstance(x, int):
        from keops.python_engine.formulas.variables.IntCst import IntCst
        return IntCst(x)
    elif isinstance(x, Operation):
        return x
    else:
        raise ValueError("invalid type : ", type(x))


##########################
#####    Broadcast    ####
##########################

# N.B. this is used internally
def Broadcast(arg, dim):
    from keops.python_engine.formulas.vectOps import SumT

    if arg.dim == dim or dim == 1:
        return arg
    elif arg.dim == 1:
        return SumT(arg, dim)
    else:
        raise ValueError("dimensions are not compatible for Broadcast operation")
