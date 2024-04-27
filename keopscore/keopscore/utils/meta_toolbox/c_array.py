from .c_instruction import c_instruction, c_empty_instruction
from .c_lvalue import c_lvalue
from .c_expression import c_expression, py2c
from .c_for import c_for_loop
from .c_variable import c_variable
from .misc import (
    Meta_Toolbox_Error,
    c_pointer_dtype,
    c_value_dtype,
    is_pointer_dtype,
    new_c_name,
)


class c_array:

    def getitem_check_convert_arg(self, other):
        if type(other) in (int, float):
            other = int(other)
            if self.dim != 0 and (other < 0 or other >= self.dim):
                Meta_Toolbox_Error("out of bound value for __getitem__")
        other = py2c(other)
        if other.dtype not in ("int", "signed long int", "float", "double"):
            Meta_Toolbox_Error(
                "v[i] with v c_array requires i.dtype='int', i.dtype='signed long int', i.dtype='float' or i.dtype='double' "
            )
        return other

    def apply(self, fun, *others):
        from .VectApply import VectApply

        return VectApply(fun, self, *others)

    @property
    def value(self):
        return self[0]


class c_array_from_address(c_array):

    def __init__(self, dim, expression):
        if dim != None and not isinstance(dim, int):
            Meta_Toolbox_Error("input dim should be None or integer")
        if dim != None and dim < 0:
            Meta_Toolbox_Error("negative dimension for array")
        if not isinstance(expression, c_expression):
            Meta_Toolbox_Error("input must be a c_expression instance")
        self.c_address = expression
        self.dim = dim
        self.dtype = c_value_dtype(expression.dtype)
        self.id = expression.code_string
        self.vars = self.c_address.vars

    def __repr__(self):
        # method for printing the c_array inside Python code
        return self.c_address.__repr__()

    def split(self, *dims):
        # split c_array in n sub arrays with dimensions dims[0], dims[1], ..., dims[n-1]
        if sum(dims) != self.dim:
            Meta_Toolbox_Error("incompatible dimensions for split")
        listarr, cumdim = [], 0
        for dim in dims:
            listarr.append(c_array_from_address(dim, self.c_address + cumdim))
            cumdim += dim
        return listarr

    def assign(self, val):
        # returns C++ code string to fill all elements of an array with a single value
        # val is a c_variable representing the value.
        loop, k = c_for_loop(0, self.dim, 1, pragma_unroll=True)
        return loop(self[k].assign(val))

    def copy(self, other):
        if not isinstance(other, c_array):
            Meta_Toolbox_Error("other should be c_array instance")
        if other.dim not in (1, self.dim):
            Meta_Toolbox_Error("incompatible dimensions for copy")
        forloop, k = c_for_loop(0, self.dim, 1, pragma_unroll=True)
        return forloop(self[k].assign(other[k]))

    def __getitem__(self, other):
        if isinstance(other, c_array_scalar):
            return self[other.c_val]
        elif isinstance(other, c_array):
            return self[other.value]
        other = self.getitem_check_convert_arg(other)
        if other.dtype in ("int", "signed long int"):
            string = f"{self.id}[{other.id}]"
        elif other.dtype == "float":
            string = f"{self.id}[(int){other.id}]"
        elif other.dtype == "double":
            string = f"{self.id}[(signed long int){other.id}]"
        vars = self.c_address.vars.union(other.vars)
        return c_lvalue(
            string_id=string, vars=vars, dtype=self.dtype, add_parenthesis=False
        )

    @property
    def c_print(self):
        if self.dtype in ["float", "double"]:
            tag = "%f, " * self.dim
        elif self.dtype in ["int", "signed long int", "float*", "double*"]:
            tag = "%d, " * self.dim
        else:
            Meta_Toolbox_Error(f"c_print not implemented for dtype={self.dtype}")
        string = f'printf("{self.id} = {tag}\\n"'
        for i in range(self.dim):
            string += f", {self[i].id}"
        string += ");\n"
        return string


class c_fixed_size_array_proper(c_array_from_address):

    def __init__(self, dtype, dim, string_id=None, qualifier=None):
        if string_id is None:
            string_id = new_c_name("array")
        expression = c_variable(c_pointer_dtype(dtype), string_id)
        super().__init__(dim, expression)
        if qualifier != None:
            self.declaration_string = qualifier + " " + dtype
        else:
            self.declaration_string = dtype
        self.qualifier = qualifier

    def declare(self, force_declare=False, **kwargs):
        # returns C++ code to declare a fixed-size arry of size dim,
        # skipping declaration if dim=0
        dim = self.dim
        if dim != None and dim == 0:
            if force_declare:
                dim = 1
            else:
                return c_empty_instruction
        if dim == None:
            dim_string = ""
        else:
            dim_string = str(dim)
        local_vars = self.vars
        global_vars = set()
        return c_instruction(
            f"{self.declaration_string} {self.c_address}[{dim_string}]",
            local_vars,
            global_vars,
            **kwargs,
        )


class c_array_variable(c_array_from_address):

    def __init__(self, dtype, string_id=None, qualifier=None):
        if string_id is None:
            string_id = new_c_name("var")
        self.c_var = c_variable(dtype, string_id)
        super().__init__(1, self.c_var.reference)
        if qualifier != None:
            self.declaration_string = qualifier + " " + dtype
        else:
            self.declaration_string = dtype
        self.qualifier = qualifier

    def __repr__(self):
        # method for printing the c_array inside Python code
        return self.c_var.__repr__()

    def declare(self, **kwargs):
        # returns C++ code to declare the variable
        return self.c_var.declare(**kwargs)

    def __getitem__(self, other):
        # N.B. we ignore other and output self.c_var
        # as if other=0. This allows broadcasting in apply and copy methods.
        return self.c_var


class c_array_scalar(c_array):

    def __init__(self, val):
        c_val = py2c(val)
        self.dtype = c_val.dtype
        self.dim = 1
        self.c_val = c_val

    def __repr__(self):
        # method for printing the c_array inside Python code
        return self.c_val.__repr__()

    def __getitem__(self, other):
        # N.B. we ignore other and output self.c_val
        # as if other=0. This allows broadcasting in apply and copy methods.
        return self.c_val


def c_fixed_size_array(dtype, dim, string_id=None, qualifier=None):
    if dim == 1:
        return c_array_variable(dtype, string_id, qualifier)
    else:
        return c_fixed_size_array_proper(dtype, dim, string_id, qualifier)
