from .c_code import c_code
from .misc import (
    c_pointer_dtype,
    c_value_dtype,
    is_pointer_dtype,
    registered_dtypes,
    Meta_Toolbox_Error,
)


class c_expression(c_code):

    def __init__(self, string, vars, dtype, add_parenthesis=True):
        if not isinstance(string, str):
            Meta_Toolbox_Error("invalid expression")
        if dtype not in registered_dtypes:
            raise ValueError(f"data type {dtype} not registered")
        self.dtype = dtype  # dtype is C++ type of variable
        super().__init__(string, vars)
        self.code_string_no_parenthesis = str(string)
        self.code_string = f"({string})" if add_parenthesis else str(string)
        self.id = self.code_string
        self.dim = 1

    def binary_op(self, other, python_op, c_op, name, dtype=None):
        other = py2c(other)
        if self.dtype != other.dtype:
            if (
                self.dtype == "int"
                and other.dtype == "signed long int"
                or self.dtype == "signed long int"
                and other.dtype == "int"
            ):
                if dtype is None:
                    dtype = "signed long int"
            elif is_pointer_dtype(self.dtype) and other.dtype in (
                "int",
                "signed long int",
            ):
                dtype = self.dtype
            else:
                Meta_Toolbox_Error(
                    f"{name} of two c_expression is only possible with same dtype, received {self.dtype} and {other.dtype}"
                )
        if dtype is None:
            dtype = self.dtype
        return c_expression(
            f"{self.code_string}{c_op}{other.code_string}",
            self.vars.union(other.vars),
            dtype,
        )

    def __add__(self, other):
        if other == 0:
            return self
        python_op = lambda x, y: x + y
        return self.binary_op(other, python_op, "+", "addition")

    def __mul__(self, other):
        if other == 1:
            return self
        elif other == 0:
            return 0
        python_op = lambda x, y: x * y
        return self.binary_op(other, python_op, "*", "product")

    def __sub__(self, other):
        if other == 0:
            return self
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
            f"{self.code_string} ? {out_true.code_string} : {out_false.code_string}",
            vars,
            out_true.dtype,
        )

    def __neg__(self):
        return c_expression(f"-{self.code_string}", self.vars, self.dtype)

    def __pow__(self, other):
        return c_expression(f"{self.code_string}**{other}", self.vars, self.dtype)

    def __getitem__(self, other):
        other = py2c(other)
        if isinstance(other, c_expression):
            if other.dtype not in ("int", "signed long int"):
                Meta_Toolbox_Error(
                    "v[i] with i and v c_expression requires i.dtype='int' or i.dtype='signed long int' "
                )
            return c_expression(
                f"{self.code_string}[{other.code_string_no_parenthesis}]",
                self.vars.union(other.vars),
                c_value_dtype(self.dtype),
                add_parenthesis=False,
            )
        else:
            Meta_Toolbox_Error("not implemented")

    @property
    def reference(self):
        return c_expression(f"&{self}", self.vars, c_pointer_dtype(self.dtype))

    @property
    def value(self):
        if is_pointer_dtype(self.dtype):
            return c_expression(f"*{self}", self.vars, c_value_dtype(self.dtype))
        else:
            return self

    def cast_to(self, dtype):
        simple_dtypes = ["float", "double", "int", "signed long int", "bool"]
        if (dtype in simple_dtypes) and (self.dtype in simple_dtypes):
            return self
        elif dtype == "half2" and self.dtype in (
            "float",
            "double",
            "int",
            "signed long int",
        ):
            string = f"__float2half2_rn({self})"
        elif dtype == "float2" and self.dtype == "half2":
            string = f"__half22float2({self})"
        elif dtype == "half2" and self.dtype == "float2":
            string = f"__float22half2_rn({self})"
        elif self.dtype in ("int", "signed long int") and is_pointer_dtype(dtype):
            string = f"{self}"
        else:
            Meta_Toolbox_Error(f"not implemented: casting from {self.dtype} to {dtype}")
        return c_expression(
            string=string, vars=self.vars, dtype=dtype, add_parenthesis=False
        )


def cast_to(dtype, expression):
    return expression.cast_to(dtype)


c_empty_expression = c_expression("", set(), "void", add_parenthesis=False)


def c_expression_from_string(string, dtype):
    # N.B. ideally we would like to suppress this function
    # to force the user to declare variables used in the code
    return c_expression(string, set(), dtype)


def py2c(expression):
    from .c_array import c_array

    if isinstance(expression, c_expression) or isinstance(expression, c_array):
        return expression
    if isinstance(expression, int):
        dtype = "signed long int" if abs(expression) > 2e9 else "int"
    elif isinstance(expression, float):
        dtype = "double"
    else:
        Meta_Toolbox_Error("invalid expression")
    return c_expression(str(expression), set(), dtype, add_parenthesis=False)


c_zero_int = c_expression("0", set(), "int", add_parenthesis=False)

c_zero_float = c_expression("0.0f", set(), "float", add_parenthesis=False)


def infinity(dtype):
    if dtype == "float":
        code = "1.0f/0.0f"
    elif dtype == "double":
        code = "1.0/0.0"
    else:
        Meta_Toolbox_Error("only float and double dtypes are implemented")
    return c_expression(code, set(), dtype)


def neg_infinity(dtype):
    return c_expression(f"-({infinity(dtype).id})", set(), dtype)
