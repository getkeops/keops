from .c_code import c_code
from .misc import is_pointer, registered_dtypes, Meta_Toolbox_Error


class c_expression(c_code):

    def __init__(self, string, vars, dtype, add_parenthesis=True):
        if not isinstance(string, str):
            Meta_Toolbox_Error("invalid expression")
        if dtype not in registered_dtypes:
            raise ValueError(f"data type {dtype} not registered")
        self.dtype = dtype  # dtype is C++ type of variable
        super().__init__(
            f"({string})" if add_parenthesis else str(string), vars
        )
        self.id = self.code_string

    def binary_op(self, other, python_op, c_op, name, dtype=None):
        other = py2c(other)
        if isinstance(other, c_expression):
            if self.dtype != other.dtype:
                if (
                    self.dtype == "int"
                    and other.dtype == "signed long int"
                    or self.dtype == "signed long int"
                    and other.dtype == "int"
                ):
                    if dtype is None:
                        dtype = "signed long int"
                else:
                    Meta_Toolbox_Error(
                        f"{name} of two c_expression is only possible with same dtype"
                    )
            if dtype is None:
                dtype = self.dtype
            return c_expression(
                f"{self.code_string}{c_op}{other.code_string}",
                self.vars.union(other.vars),
                dtype,
            )
        else:
            Meta_Toolbox_Error("not implemented")

    def __add__(self, other):
        python_op = lambda x, y: x + y
        return self.binary_op(other, python_op, "+", "addition")

    def __mul__(self, other):
        python_op = lambda x, y: x * y
        return self.binary_op(other, python_op, "*", "product")

    def __sub__(self, other):
        python_op = lambda x, y: x - y
        return self.binary_op(other, python_op, "-", "subtraction")

    def __truediv__(self, other):
        python_op = lambda x, y: x / y
        return self.binary_op(other, python_op, "/", "division")

    def __lt__(self, other):
        python_op = lambda x, y: x < y
        return self.binary_op(other, python_op, "<", "comparison", dtype="bool")

    def __le__(self, other):
        python_op = lambda x, y: x <= y
        return self.binary_op(other, python_op, "<=", "comparison", dtype="bool")

    def __gt__(self, other):
        python_op = lambda x, y: x > y
        return self.binary_op(other, python_op, ">", "comparison", dtype="bool")

    def __ge__(self, other):
        python_op = lambda x, y: x >= y
        return self.binary_op(other, python_op, ">=", "comparison", dtype="bool")

    # N.B.: The & symbol (__and__) denotes the bitwise operator, not the logical one
    def logical_and(self, other):
        python_op = lambda x, y: x and y
        return self.binary_op(other, python_op, "&&", "comparison", dtype="bool")

    # N.B.: The | symbol (__or__) denotes the bitwise operator, not the logical one
    def logical_or(self, other):
        python_op = lambda x, y: x or y
        return self.binary_op(other, python_op, "||", "comparison", dtype="bool")

    def ternary(self, out_true, out_false):
        if self.dtype != "bool":
            Meta_Toolbox_Error(
                f"The input of a ternary operator should have bool dtype, "
                f"found {self.dtype}."
            )
        if out_true.dtype != out_false.dtype:
            Meta_Toolbox_Error(
                f"The two possible output of a ternary operator should have the same dtype, "
                f"found {out_true.dtype} and {out_false.dtype}."
            )
        vars = self.vars.union(out_true.vars.union(out_false.vars))
        return c_expression(
            f"({self.code_string} ? {out_true.code_string} : {out_false.code_string})",
            vars,
            out_true.dtype,
        )

    def __neg__(self):
        return c_expression(f"(-{self.code})", self.vars, self.dtype)

    def __getitem__(self, other):
        other = py2c(other)
        if isinstance(other, c_expression):
            if other.dtype not in ("int", "signed long int"):
                Meta_Toolbox_Error(
                    "v[i] with i and v c_expression requires i.dtype='int' or i.dtype='signed long int' "
                )
            return c_expression(
                f"{self.code_string}[{other.code_string}]",
                self.vars.union(other.vars),
                c_value(self.dtype),
            )
        else:
            Meta_Toolbox_Error("not implemented")

c_empty_expression = c_expression("", set(), "void", add_parenthesis=False)

def py2c(expression):
    if isinstance(expression, c_expression):
        return expression
    if isinstance(expression, int):
        dtype = "signed long int" if expression > 2e9 else "int"
    elif isinstance(expression, float):
        dtype = "double"
    else:
        Meta_Toolbox_Error("invalid expression")
    return c_expression(str(expression), set(), dtype, add_parenthesis=False)


class cast_to(c_expression):

    def __init__(self, dtype, expr):
        simple_dtypes = ["float", "double", "int", "signed long int", "bool"]
        if (dtype in simple_dtypes) and (expr.dtype in simple_dtypes):
            string = f"({dtype}){expr}"
        elif dtype == "half2" and expr.dtype == "float":
            string = f"__float2half2_rn({expr})"
        elif dtype == "float2" and expr.dtype == "half2":
            string = f"__half22float2({expr})"
        elif dtype == "half2" and expr.dtype == "float2":
            string = f"__float22half2_rn({expr})"
        elif expr.dtype in ("int", "signed long int") and is_pointer(dtype):
            string = f"{expr}"
        else:
            Meta_Toolbox_Error(f"not implemented: casting from {expr.dtype} to {dtype}")
        super().__init__(string=string, vars=expr.vars, dtype=dtype, add_parenthesis=False)


def c_value(x):
    # either convert c_array or c_variable representing a pointer to its value c_variable (dereference)
    # or converts string "dtype*" to "dtype"
    from .c_array import c_array

    if isinstance(x, c_array):
        return c_expression(f"(*{x})", vars=x.vars, dtype=x.dtype)
    if isinstance(x, c_expression):
        return c_expression(c_value(f"(*{x})", vars=x.vars, dtype=x.dtype))
    elif isinstance(x, str):
        if x[-1] != "*":
            Meta_Toolbox_Error("dtype is not a pointer type")
        return x[:-1]
    else:
        Meta_Toolbox_Error("input should be c_expression instance or string.")


def c_pointer(x):
    # either convert c_expression to its address c_expression (reference)
    # or converts string "dtype" to "dtype*"
    if isinstance(x, c_expression):
        return c_expression(f"(&{x})", x.vars, c_pointer(x.dtype))
    elif isinstance(x, str):
        return x + "*"
    else:
        Meta_Toolbox_Error("input should be either c_variable instance or string.")
