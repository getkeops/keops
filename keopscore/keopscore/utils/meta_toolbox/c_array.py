from ctypes import pointer
from c_expression import c_pointer, py2c
from c_for import c_for_loop
from c_variable import c_variable
from misc import Error, new_c_name


class c_array:
    def __init__(self, dtype, dim, string_id=new_c_name("array")):
        if dim != "" and dim < 0:
            Error("negative dimension for array")
        self.c_var = c_variable(c_pointer(dtype), string_id)
        self.dtype = dtype
        self.dim = dim
        self.id = string_id

    def __repr__(self):
        # method for printing the c_variable inside Python code
        return self.c_var.__repr__()

    def declare(self):
        # returns C++ code to declare a fixed-size arry of size dim,
        # skipping declaration if dim=0
        if self.dim == "" or self.dim > 0:
            return f"{self.dtype} {self.c_var}[{self.dim}];"
        else:
            return ""

    def split(self, *dims):
        # split c_array in n sub arrays with dimensions dims[0], dims[1], ..., dims[n-1]
        if sum(dims) != self.dim:
            Error("incompatible dimensions for split")
        listarr, cumdim = [], 0
        for dim in dims:
            listarr.append(c_array(self.dtype, dim, f"({self.id}+{cumdim})"))
            cumdim += dim
        return listarr

    def assign(self, val):
        # returns C++ code string to fill all elements of a fixed size array with a single value
        # val is a c_variable representing the value.
        loop, k = c_for_loop(0, self.dim, 1, pragma_unroll=True)
        return loop(self[k].assign(val))

    def __getitem__(self, other):
        other = py2c(other)
        if isinstance(other, c_variable):
            if other.dtype in ("int", "signed long int"):
                return c_variable(self.dtype, f"{self.id}[{other.id}]")
            elif other.dtype == "float":
                return c_variable(self.dtype, f"{self.id}[(int){other.id}]")
            elif other.dtype == "double":
                return c_variable(self.dtype, f"{self.id}[(signed long int){other.id}]")
            else:
                Error(
                    "v[i] with i and v c_array requires i.dtype='int', i.dtype='signed long int', i.dtype='float' or i.dtype='double' "
                )
        else:
            Error("not implemented")

    @property
    def c_print(self):
        if self.dtype in ["float", "double"]:
            tag = "%f, " * self.dim
        elif self.dtype in ["int", "signed long int", "float*", "double*"]:
            tag = "%d, " * self.dim
        else:
            Error(f"c_print not implemented for dtype={self.dtype}")
        string = f'printf("{self.id} = {tag}\\n"'
        for i in range(self.dim):
            string += f", {self[i].id}"
        string += ");\n"
        return string
