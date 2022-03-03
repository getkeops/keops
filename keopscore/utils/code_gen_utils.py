import os
from hashlib import sha256

from keopscore.config.config import disable_pragma_unrolls
from keopscore.utils.misc_utils import KeOps_Error, KeOps_Message


def get_hash_name(*args):
    return sha256("".join(list(str(arg) for arg in args)).encode("utf-8")).hexdigest()[
        :10
    ]


#######################################################################
# .  Python to C++ meta programming toolbox
#######################################################################


def sizeof(dtype):
    if dtype == "float":
        return 4
    elif dtype == "double":
        return 8
    elif dtype == "half":
        return 2
    else:
        KeOps_Error("not implemented")


class new_c_varname:
    # class to generate unique names for variables in C++ code, to avoid conflicts
    dict_instances = {}

    def __new__(self, template_string_id, num=1, as_list=False):
        # - template_string_id is a string, the base name for c_variable
        # - if num>1 returns a list of num new names with same base names
        # For example the first call to new_c_variable("x")
        # will return "x_1", the second call will return "x_2", etc.
        if num > 1 or as_list:
            return list(new_c_varname(template_string_id) for k in range(num))
        if template_string_id in new_c_varname.dict_instances:
            cnt = new_c_varname.dict_instances[template_string_id] + 1
        else:
            cnt = 0
        new_c_varname.dict_instances[template_string_id] = cnt
        string_id = template_string_id + "_" + str(cnt)
        return string_id


class c_variable:
    # class to represent a C++ variable, storing its c++ name and its C++ type.
    def __new__(self, dtype, list_string_id=None):
        if isinstance(list_string_id, list):
            return list(c_variable(dtype, string_id) for string_id in list_string_id)
        else:
            return super(c_variable, self).__new__(self)

    def __init__(self, dtype, string_id=None):
        if string_id is None:
            string_id = new_c_varname("var")
        self.dtype = dtype  # dtype is C++ type of variable
        self.id = string_id  # string_id is C++ name of variable

    def __repr__(self):
        # method for printing the c_variable inside Python code
        return self.id

    def declare(self):
        return f"{self.dtype} {self.id};\n"

    def declare_assign(self, value):
        return f"{self.dtype} " + self.assign(value)

    def assign(self, value):
        if type(value) in (int, float):
            dtype = "int" if type(value) == int else "float"
            return self.assign(c_variable(dtype, str(value)))
        elif type(value) == str:
            return f"{self.id} = ({self.dtype})({value});\n"
        elif value.dtype != self.dtype:
            if self.dtype == "float2" and value.dtype == "float":
                return f"""
                            {self.id}.x = {value.id};
                            {self.id}.y = {value.id};
                        """
            else:
                return f"{self.id} = {cast_to(self.dtype, value)};\n"
        else:
            return f"{self.id} = ({value.id});\n"

    def add_assign(self, value):
        if type(value) in (int, float):
            dtype = "int" if type(value) == int else "float"
            return self.add_assign(c_variable(dtype, str(value)))
        if type(value) == str:
            return f"{self.id} += ({self.dtype})({value});\n"
        elif value.dtype != self.dtype:
            if self.dtype == "float2" and value.dtype == "half2":
                return f"""
                            {self.id}.x += (float){value.id}.x;
                            {self.id}.y += (float){value.id}.y;
                        """
            else:
                return f"{self.id} += {cast_to(self.dtype, value)};\n"
        else:
            return f"{self.id} += ({value.id});\n"

    def __add__(self, other):
        if type(other) in (int, float):
            dtype = "int" if type(other) == int else "float"
            return self + c_variable(dtype, str(other))
        elif type(other) == c_variable:
            if self.dtype != other.dtype:
                KeOps_Error("addition of two c_variable only possible with same dtype")
            return c_variable(self.dtype, f"({self.id}+{other.id})")
        else:
            KeOps_Error("not implemented")

    def __mul__(self, other):
        if type(other) in (int, float):
            dtype = "int" if type(other) == int else "float"
            return self * c_variable(dtype, str(other))
        elif type(other) == c_variable:
            if self.dtype != other.dtype:
                KeOps_Error(
                    "multiplication of two c_variable only possible with same dtype"
                )
            return c_variable(self.dtype, f"({self.id}*{other.id})")
        else:
            KeOps_Error("not implemented")

    def __sub__(self, other):
        if type(other) in (int, float):
            dtype = "int" if type(other) == int else "float"
            return self - c_variable(dtype, str(other))
        elif type(other) == c_variable:
            if self.dtype != other.dtype:
                KeOps_Error(
                    "subtraction of two c_variable only possible with same dtype"
                )
            return c_variable(self.dtype, f"({self.id}-{other.id})")
        else:
            KeOps_Error("not implemented")

    def __truediv__(self, other):
        if type(other) in (int, float):
            dtype = "int" if type(other) == int else "float"
            return self / c_variable(dtype, str(other))
        elif type(other) == c_variable:
            if self.dtype != other.dtype:
                KeOps_Error("division of two c_variable only possible with same dtype")
            return c_variable(self.dtype, f"({self.id}/{other.id})")
        else:
            KeOps_Error("not implemented")

    def __lt__(self, other):
        if type(other) in (int, float):
            dtype = "int" if type(other) == int else "float"
            return self < c_variable(dtype, str(other))
        elif type(other) == c_variable:
            if self.dtype != other.dtype:
                KeOps_Error(
                    "comparison of two c_variable only possible with same dtype"
                )
            return c_variable("bool", f"({self.id}<{other.id})")
        else:
            KeOps_Error("not implemented")

    def __gt__(self, other):
        if type(other) in (int, float):
            dtype = "int" if type(other) == int else "float"
            return self > c_variable(dtype, str(other))
        elif type(other) == c_variable:
            if self.dtype != other.dtype:
                KeOps_Error(
                    "comparison of two c_variable only possible with same dtype"
                )
            return c_variable("bool", f"({self.id}>{other.id})")
        else:
            KeOps_Error("not implemented")

    def __neg__(self):
        return c_variable(self.dtype, f"(-{self.id})")

    def __getitem__(self, other):
        if type(other) == int:
            return self[c_variable("int", str(other))]
        elif type(other) == c_variable:
            if other.dtype != "int":
                KeOps_Error("v[i] with i and v c_variable requires i.dtype='int' ")
            return c_variable(value(self.dtype), f"{self.id}[{other.id}]")
        else:
            KeOps_Error("not implemented")


def use_pragma_unroll(n=64):
    if disable_pragma_unrolls:
        return "\n"
    else:
        if n is None:
            return f"\n#pragma unroll\n"
        else:
            return f"\n#pragma unroll({n})\n"


def c_for_loop(start, end, incr, pragma_unroll=False):
    def to_string(x):
        if type(x) == c_variable:
            if x.dtype != "int":
                KeOps_Error("only simple int type for loops implemented")
            return x.id
        elif type(x) == int:
            return str(x)
        else:
            KeOps_Error("only simple int type for loops implemented")

    start, end, incr = map(to_string, (start, end, incr))
    k = c_variable("int", new_c_varname("k"))

    def printfun(body_code):
        string = ""
        if pragma_unroll:
            string += use_pragma_unroll()
        string += f""" for(int {k.id}={start}; {k.id}<{end}; {k.id}+=({incr})) {{
                            {body_code}
                        }}
                    """
        return string

    return printfun, k


c_zero_int = c_variable("int", "0")
c_zero_float = c_variable("float", "0.0f")


def neg_infinity(dtype):
    return c_variable(dtype, f"-({infinity(dtype).id})")


def infinity(dtype):
    if dtype == "float":
        code = "( 1.0f/0.0f )"
    elif dtype == "double":
        code = "( 1.0/0.0 )"
    else:
        KeOps_Error(
            "only float and double dtypes are implemented in new python engine for now"
        )
    return c_variable(dtype, code)


def cast_to(dtype, var):
    # returns C++ code string to do a cast ; e.g. "(float)" if dtype is "float" for example
    simple_dtypes = ["float", "double", "int"]
    if (dtype in simple_dtypes) and (var.dtype in simple_dtypes):
        return f"({dtype})({var.id})"
    elif dtype == "half2" and var.dtype == "float":
        return f"__float2half2_rn({var.id})"
    elif dtype == "float2" and var.dtype == "half2":
        return f"__half22float2({var.id})"
    elif dtype == "half2" and var.dtype == "float2":
        return f"__float22half2_rn({var.id})"
    else:
        KeOps_Error(f"not implemented: casting from {var.dtype} to {dtype}")


def value(x):
    # either convert c_array or c_variable representing a pointer to its value c_variable (dereference)
    # or converts string "dtype*" to "dtype"
    if isinstance(x, c_array):
        return c_variable(x.dtype, f"(*{x.id})")
    if isinstance(x, c_variable):
        return c_variable(value(x.dtype), f"(*{x.id})")
    elif isinstance(x, str):
        if x[-1] == "*":
            return x[:-1]
        else:
            KeOps_Error(
                "Incorrect input string in value function; it should represent a pointer C++ type."
            )
    else:
        KeOps_Error("input should be either c_variable instance or string.")


def pointer(x):
    # either convert c_variable to its address c_variable (reference)
    # or converts string "dtype" to "dtype*"
    if isinstance(x, c_variable):
        return c_variable(pointer(x.dtype), f"(&{x.id})")
    elif isinstance(x, str):
        return x + "*"
    else:
        KeOps_Error("input should be either c_variable instance or string.")


class c_array:
    def __init__(self, dtype, dim, string_id=new_c_varname("array")):
        if dim < 0:
            KeOps_Error("negative dimension for array")
        self.c_var = c_variable(pointer(dtype), string_id)
        self.dtype = dtype
        self.dim = dim
        self.id = string_id

    def __repr__(self):
        # method for printing the c_variable inside Python code
        return self.c_var.__repr__()

    def declare(self):
        # returns C++ code to declare a fixed-size arry of size dim,
        # skipping declaration if dim=0
        if self.dim > 0:
            return f"{self.dtype} {self.c_var.id}[{self.dim}];"
        else:
            return ""

    def split(self, *dims):
        # split c_array in n sub arrays with dimensions dims[0], dims[1], ..., dims[n-1]
        if sum(dims) != self.dim:
            KeOps_Error("incompatible dimensions for split")
        listarr, cumdim = [], 0
        for dim in dims:
            listarr.append(c_array(self.dtype, dim, f"({self.id}+{cumdim})"))
            cumdim += dim
        return listarr

    def assign(self, val):
        # returns C++ code string to fill all elements of a fixed size array with a single value
        # val is a c_variable representing the value.
        loop, k = c_for_loop(0, self.dim, 1)
        return loop(self[k].assign(val))

    def __getitem__(self, other):
        if type(other) == int:
            return self[c_variable("int", str(other))]
        elif type(other) == c_variable:
            if other.dtype != "int":
                KeOps_Error("v[i] with i and v c_array requires i.dtype='int' ")
            return c_variable(self.dtype, f"{self.id}[{other.id}]")
        else:
            KeOps_Error("not implemented")

    @property
    def c_print(self):
        if self.dtype in ["float", "double"]:
            tag = "%f, " * self.dim
        elif self.dtype in ["int", "float*", "double*"]:
            tag = "%d, " * self.dim
        else:
            KeOps_Error(f"c_print not implemented for dtype={self.dtype}")
        string = f'printf("{self.id} = {tag}\\n"'
        for i in range(self.dim):
            string += f", {self[i].id}"
        string += ");\n"
        return string


def VectApply(fun, out, *args):
    # returns C++ code string to apply a scalar operation to fixed-size arrays, following broadcasting rules.
    # - fun is the scalar unary function to be applied, it must accept two c_variable or c_array inputs and output a string
    # - out must be a c_array instance
    # - args may be c_array or c_variable instances
    #
    # Example : if out.dim = 3, arg0.dim = 1, arg1.dim = 3,
    # it will generate the following (in pseudo-code for clarity) :
    #   #pragma unroll
    #   for(int k=0; k<out.dim; k++)
    #       fun(out[k], arg0[0], arg1[k]);
    #
    # Equivalently, if out.dim = 3, arg0 is c_variable, arg1.dim = 3,
    # it will generate the following (in pseudo-code for clarity) :
    #   #pragma unroll
    #   for(int k=0; k<out.dim; k++)
    #       fun(out[k], arg0, arg1[k]);

    dims = [out.dim]
    for arg in args:
        if isinstance(arg, c_variable):
            dims.append(1)
        elif isinstance(arg, c_array):
            dims.append(arg.dim)
        else:
            KeOps_Error("args must be c_variable or c_array instances")
    dimloop = max(dims)
    if not set(dims) in ({dimloop}, {1, dimloop}):
        KeOps_Error("incompatible dimensions in VectApply")
    incr_out = 1 if out.dim == dimloop else 0
    incr_args = list((1 if dim == dimloop else 0) for dim in dims[1:])

    forloop, k = c_for_loop(0, dimloop, 1, pragma_unroll=True)

    argsk = []
    for (arg, incr) in zip(args, incr_args):
        if isinstance(arg, c_variable):
            argsk.append(arg)
        elif isinstance(arg, c_array):
            argsk.append(arg[k * incr])

    return forloop(fun(out[k * incr_out], *argsk))


def ComplexVectApply(fun, out, *args):
    # similar to VectApply but for complex operations

    dims = [out.dim]
    for arg in args:
        if isinstance(arg, c_array):
            dims.append(arg.dim)
        else:
            KeOps_Error("args must be c_array instances")
    dimloop = max(dims)
    if not set(dims) in ({dimloop}, {2, dimloop}):
        KeOps_Error("incompatible dimensions in ComplexVectApply")
    incr_out = 1 if out.dim == dimloop else 0
    incr_args = list((1 if dim == dimloop else 0) for dim in dims[1:])

    forloop, k = c_for_loop(0, dimloop, 2, pragma_unroll=True)

    argsk = []
    for (arg, incr) in zip(args, incr_args):
        argk = c_array(arg.dtype, 2, f"({arg.id}+{k.id}*{incr})")
        argsk.append(argk)
    outk = c_array(out.dtype, 2, f"({out.id}+{k.id}*{incr_out})")
    return forloop(fun(outk, *argsk))


def VectCopy(out, arg, dim=None):
    # returns a C++ code string representing a vector copy between fixed-size arrays
    # - dim is dimension of arrays
    # - out is c_variable representing the output array
    # - arg is c_variable representing the input array
    if dim is None:
        dim = out.dim
    forloop, k = c_for_loop(0, dim, 1, pragma_unroll=True)
    return forloop(out[k].assign(arg[k]))


def call_list(args):
    return ", ".join(list(arg.id for arg in args))


def signature_list(args):
    return ", ".join(list(f"{arg.dtype} {arg.id}" for arg in args))


def c_include(*headers):
    return "".join(f"#include <{header}>\n" for header in headers)


def c_if(condition, command, else_command=None):
    string = f""" if ({condition.id}) {{
                      {command}
                }}
            """
    if else_command:
        string += f""" else {{
                      {else_command}
                }}
            """
    return string


def c_block(*commands):
    block_string = "".join(commands)
    return f""" {{
                      {block_string}
                }}
            """


def c_function(name, dtypeout, args, commands, qualifier=None):
    # first write the signature of the function :
    string = ""
    if qualifier is not None:
        string += f"{qualifier} "
    string += f"{dtypeout} {name}({signature_list(args)}) "
    # then the body
    string += "\n{\n"
    string += "\n".join(list(c for c in commands))
    string += "\n}\n"
    return string


#######################################################################
# .  KeOps related helpers
#######################################################################


def GetDims(Vars):
    # returns the list of dim fields (dimensions) of a list of Var instances
    return tuple(v.dim for v in Vars)


def GetInds(Vars):
    # returns the list of ind fields (indices) of a list of Var instances
    return tuple(v.ind for v in Vars)


class Var_loader:
    def __init__(self, red_formula):

        formula = red_formula.formula
        tagI, tagJ = red_formula.tagI, red_formula.tagJ

        mymin = lambda x: min(x) if len(x) > 0 else -1

        self.Varsi = formula.Vars(
            cat=tagI
        )  # list all "i"-indexed variables in the formula
        self.nvarsi = len(self.Varsi)  # number of "i"-indexed variables
        self.indsi = GetInds(self.Varsi)  # list indices of "i"-indexed variables
        self.pos_first_argI = mymin(self.indsi)  # first index of "i"-indexed variables
        self.dimsx = GetDims(self.Varsi)  # list dimensions of "i"-indexed variables
        self.dimx = sum(self.dimsx)  # total dimension of "i"-indexed variables

        self.Varsj = formula.Vars(
            cat=tagJ
        )  # list all "j"-indexed variables in the formula
        self.nvarsj = len(self.Varsj)  # number of "j"-indexed variables
        self.indsj = GetInds(self.Varsj)  # list indices of "j"-indexed variables
        self.pos_first_argJ = mymin(self.indsj)  # first index of "j"-indexed variables
        self.dimsy = GetDims(self.Varsj)  # list dimensions of "j"-indexed variables
        self.dimy = sum(self.dimsy)  # total dimension of "j"-indexed variables

        self.Varsp = formula.Vars(cat=2)  # list all parameter variables in the formula
        self.nvarsp = len(self.Varsp)  # number of parameter variables
        self.indsp = GetInds(self.Varsp)  # list indices of parameter variables
        self.pos_first_argP = mymin(self.indsp)  # first index of parameter variables
        self.dimsp = GetDims(self.Varsp)  # list indices of parameter variables
        self.dimp = sum(self.dimsp)  # total dimension of parameter variables

        self.inds = GetInds(formula.Vars_)
        self.nminargs = max(self.inds) + 1 if len(self.inds) > 0 else 0

    def table(self, xi, yj, pp):
        return table(
            self.nminargs,
            self.dimsx,
            self.dimsy,
            self.dimsp,
            self.indsi,
            self.indsj,
            self.indsp,
            xi,
            yj,
            pp,
        )

    def direct_table(self, args, i, j):
        return direct_table(
            self.nminargs,
            self.dimsx,
            self.dimsy,
            self.dimsp,
            self.indsi,
            self.indsj,
            self.indsp,
            args,
            i,
            j,
        )

    def load_vars(self, cat, *args, **kwargs):
        if cat == "i":
            dims, inds = self.dimsx, self.indsi
        elif cat == "j":
            dims, inds = self.dimsy, self.indsj
        elif cat == "p":
            dims, inds = self.dimsp, self.indsp
        return load_vars(dims, inds, *args, **kwargs)


def table(nminargs, dimsx, dimsy, dimsp, indsi, indsj, indsp, xi, yj, pp):
    res = [None] * nminargs
    for (dims, inds, xloc) in (
        (dimsx, indsi, xi),
        (dimsy, indsj, yj),
        (dimsp, indsp, pp),
    ):
        k = 0
        for u in range(len(dims)):
            res[inds[u]] = c_array(xloc.dtype, dims[u], f"({xloc.id}+{k})")
            k += dims[u]
    return res


def direct_table(nminargs, dimsx, dimsy, dimsp, indsi, indsj, indsp, args, i, j):
    res = [None] * nminargs
    for (dims, inds, row_index) in (
        (dimsx, indsi, i),
        (dimsy, indsj, j),
        (dimsp, indsp, c_zero_int),
    ):
        for u in range(len(dims)):
            arg = args[inds[u]]
            res[inds[u]] = c_array(
                value(arg.dtype), dims[u], f"({arg.id}+{row_index.id}*{dims[u]})"
            )
    return res


def table4(
    nminargs,
    dimsx,
    dimsy,
    dimsp,
    dims_new,
    indsi,
    indsj,
    indsp,
    inds_new,
    xi,
    yj,
    pp,
    arg_new,
):
    res = [None] * nminargs
    for (dims, inds, xloc) in (
        (dimsx, indsi, xi),
        (dimsy, indsj, yj),
        (dimsp, indsp, pp),
        (dims_new, inds_new, arg_new),
    ):
        k = 0
        for u in range(len(dims)):
            res[inds[u]] = c_array(xloc.dtype, dims[u], f"({xloc.id}+{k})")
            k += dims[u]

    return res


def load_vars(dims, inds, xloc, args, row_index=c_zero_int, offsets=None, indsref=None):
    # returns a c++ code used to create a local copy of slices of the input tensors, for evaluating a formula
    # - dims is a list of integers giving dimensions of variables
    # - dims is a list of integers giving indices of variables
    # - xloc is a c_array, the local array which will receive the copy
    # - args is a list of c_variable, representing pointers to input tensors
    # - row_index is a c_variable (of dtype="int"), specifying which row of the matrix should be loaded
    # - offsets is an optional c_array (of dtype="int"), specifying variable-dependent offsets (used when broadcasting batch dimensions)
    # - indsref is an optional list of integers, giving index mapping for offsets
    #
    # Example: assuming i=c_variable("int", "5"), xloc=c_variable("float", "xi") and px=c_variable("float**", "px"), then
    # if dims = [2,2,3] and inds = [7,9,8], the call to
    #   load_vars (dims, inds, xi, [arg0, arg1,..., arg9], row_index=i )
    # will output the following code:
    #   xi[0] = arg7[5*2+0];
    #   xi[1] = arg7[5*2+1];
    #   xi[2] = arg9[5*2+0];
    #   xi[3] = arg9[5*2+1];
    #   xi[4] = arg8[5*3+0];
    #   xi[5] = arg8[5*3+1];
    #   xi[6] = arg8[5*3+2];
    #
    # Example (with offsets): assuming i=c_variable("int", "5"),
    # xloc=c_variable("float", "xi"), px=c_variable("float**", "px"),
    # and offsets = c_array("int", 3, "offsets"), then
    # if dims = [2,2,3] and inds = [7,9,8], the call to
    #   load_vars (dims, inds, xi, [arg0, arg1,..., arg9], row_index=i, offsets=offsets)
    # will output the following code:
    #   xi[0] = arg7[(5+offsets[0])*2+0];
    #   xi[1] = arg7[(5+offsets[0])*2+1];
    #   xi[2] = arg9[(5+offsets[1])*2+0];
    #   xi[3] = arg9[(5+offsets[1])*2+1];
    #   xi[4] = arg8[(5+offsets[2])*3+0];
    #   xi[5] = arg8[(5+offsets[2])*3+1];
    #   xi[6] = arg8[(5+offsets[2])*3+2];
    #
    # Example (with offsets and indsref): assuming i=c_variable("int", "5"),
    # xloc=c_variable("float", "xi"), px=c_variable("float**", "px"),
    # offsets = c_array("int", 3, "offsets"),
    # if dims = [2,2,3] and inds = [7,9,8],
    # and indsref = [8,1,7,3,9,2], then since 7,8,9 are at positions 2,0,4 in indsref,
    # the call to
    #   load_vars (dims, inds, xi, [arg0, arg1,..., arg9], row_index=i, offsets=offsets, indsref=indsref)
    # will output the following code:
    #   xi[0] = arg7[(5+offsets[2])*2+0];
    #   xi[1] = arg7[(5+offsets[2])*2+1];
    #   xi[2] = arg9[(5+offsets[0])*2+0];
    #   xi[3] = arg9[(5+offsets[0])*2+1];
    #   xi[4] = arg8[(5+offsets[4])*3+0];
    #   xi[5] = arg8[(5+offsets[4])*3+1];
    #   xi[6] = arg8[(5+offsets[4])*3+2];
    string = ""
    if len(dims) > 0:
        string += "{\n"
        string += "int a=0;\n"
        for u in range(len(dims)):
            l = indsref.index(inds[u]) if indsref else u
            row_index_str = (
                f"({row_index.id}+{offsets.id}[{l}])" if offsets else row_index.id
            )
            string += use_pragma_unroll()
            string += f"for(int v=0; v<{dims[u]}; v++) {{\n"
            string += (
                f"    {xloc.id}[a] = {args[inds[u]].id}[{row_index_str}*{dims[u]}+v];\n"
            )
            string += "     a++;\n"
            string += "}\n"
        string += "}\n"
    return string


def load_vars_chunks(
    inds, dim_chunk, dim_chunk_load, dim_org, xloc, args, k, row_index=c_zero_int
):
    #
    # loads chunks of variables, unrolling dimensions and indices.
    #
    # Example:
    #   load_chunks([7,9,8], 3, 2, 11, xi, px, k, row_index=i)
    # with:
    # xi = c_variable("float", "xi"),
    # px = c_variable("float**", "px")
    # i = c_variable("int","5"),
    # k = c_variable("int","k")
    # means : there are 3 chunks of vectors to load. They are located
    # at positions 7, 9 and 8 in px. Now i=5 and dim_org=11, so we start
    # to load vectors at positions px[7]+5*11, px[9]+5*11, px[8]+5*11.
    # For each, we load the kth chunk, assuming vector is divided
    # into chunks of size 3. And finally, we stop after loading 2 values.
    #
    # So we will execute:
    #   xi[0] = px[7][5*11+k*3];
    #   xi[1] = px[7][5*11+k*3+1];
    #   xi[2] = px[9][5*11+k*3];
    #   xi[3] = px[9][5*11+k*3+1];
    #   xi[4] = px[8][5*11+k*3];
    #   xi[5] = px[8][5*11+k*3+1];
    string = ""
    if len(inds) > 0:
        string += "{"
        string += "int a=0;\n"
        for u in range(len(inds)):
            string += use_pragma_unroll()
            string += f"for(int v=0; v<{dim_chunk_load}; v++) {{\n"
            string += f"    {xloc.id}[a] = {args[inds[u]].id}[{row_index.id}*{dim_org}+{k.id}*{dim_chunk}+v];\n"
            string += "     a++;\n"
            string += "}"
        string += "}"
    return string


def load_vars_chunks_offsets(
    inds,
    indsref,
    dim_chunk,
    dim_chunk_load,
    dim_org,
    xloc,
    args,
    k,
    offsets,
    row_index=c_zero_int,
):
    # Version with variable-dependent offsets (used when broadcasting batch dimensions)
    # indsref gives mapping for offsets indexing
    # Example:
    #   load_vars_chunks_offsets([2,3,1], [8,9,7,3,1,2], 3, 2, 11, xi, px, k, offsets, row_index=i)
    # Since 2,3,1 are at positions 5,3,4 respectively in the list [8,9,7,3,1,2],
    # will output:
    #   xi[0] = px[2][(5+offsets[5])*11+k*3];
    #   xi[1] = px[2][(5+offsets[5])*11+k*3+1];
    #   xi[2] = px[3][(5+offsets[3])*11+k*3];
    #   xi[3] = px[3][(5+offsets[3])*11+k*3+1];
    #   xi[4] = px[1][(5+offsets[4])*11+k*3];
    #   xi[5] = px[1][(5+offsets[4])*11+k*3+1];
    string = ""
    if len(inds) > 0:
        string = "{"
        string += "int a=0;\n"
        for u in range(len(inds)):
            l = indsref.index(inds[u])
            string += use_pragma_unroll()
            string += f"for(int v=0; v<{dim_chunk_load}; v++) {{\n"
            string += f"    {xloc.id}[a] = {args[inds[u]].id}[({row_index.id}+{offsets.id}[{l}])*{dim_org}+{k.id}*{dim_chunk}+v];\n"
            string += "     a++;\n"
            string += "}"
        string += "}"
    return string


def varseq_to_array(vars, vars_ptr_name):
    # returns the C++ code corresponding to storing the values of a sequence of variables
    # into an array.

    dtype = vars[0].dtype
    nvars = len(vars)

    # we check that all variables have the same type
    if not all(var.dtype == dtype for var in vars[1:]):
        KeOps_Error("[KeOps] internal error ; incompatible dtypes in varseq_to_array.")
    string = f"""   {dtype} {vars_ptr_name}[{nvars}];
              """
    for i in range(nvars):
        string += f"""  {vars_ptr_name}[{i}] = {vars[i].id};
                   """
    return string


def clean_keops(recompile_jit_binary=True, verbose=True):
    import keopscore.config.config
    from keopscore.config.config import get_build_folder

    build_path = get_build_folder()
    use_cuda = keopscore.config.config.use_cuda
    if use_cuda:
        from keopscore.config.config import jit_binary
    else:
        jit_binary = None
    for f in os.scandir(build_path):
        if recompile_jit_binary or f.path != jit_binary:
            os.remove(f.path)
    if verbose:
        KeOps_Message(f"{build_path} has been cleaned.")
    from keopscore.get_keops_dll import get_keops_dll

    get_keops_dll.reset()
    if use_cuda and recompile_jit_binary:
        from keopscore.binders.nvrtc.Gpu_link_compile import Gpu_link_compile

        Gpu_link_compile.compile_jit_compile_dll()
