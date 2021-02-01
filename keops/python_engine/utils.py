
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

def call_list(args):
    return ", ".join(list(arg() for arg in args))

def signature_list(args):
    return ", ".join(list(f"{arg.dtype} {arg()}" for arg in args))

def value(pdtype):
    # converts string "dtype*" to "dtype" 
    if pdtype[-1]=="*":
        return pdtype[:-1]
    else:
        raise ValueError("Incorrect input string in value function; it should represent a pointer C++ type.")

def pointer(dtype):
    # converts string "dtype" to "dtype*" 
    return dtype+"*"
    
class new_c_varname:
    # class to generate unique names for variables in C++ code, to avoid conflicts
    dict_instances = {}
    def __new__(self, template_string_id, num=1):
        # - template_string_id is a string, the base name for c_variable
        # - dtype is a string, the C++ type of the c_variable to create
        # - if num>1 returns a list of num new variables with same base names
        # For example the first call to new_c_variable("x","float")
        # will create a c_variable with string_id="x_1", the second call 
        # will create a c_variable with string_id="x_2", etc.   
        if num > 1:
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
    def __new__(self, list_string_id, dtype):
        if isinstance(list_string_id,list):
            return list(c_variable(string_id, dtype) for string_id in list_string_id) 
        else:
            return super(c_variable, self).__new__(self)
    def __init__(self, string_id, dtype):
        self.string_id = string_id      # string_id is C++ name of variable
        self.dtype = dtype              # dtype is C++ type of variable
    def __call__(self):
        # ouputs string_id, the C++ name of the variable
        return self.string_id
    def __repr__(self):
        # method for printing the c_variable inside Python code
        return self.string_id
        
c_zero_int = c_variable("0","int")
c_zero_float = c_variable("0.0f","float")
    
class c_array:
    def __init__(self, string_id, dtype, dim):
        self.c_var = c_variable(string_id, pointer(dtype))
        self.dtype = dtype
        self.dim = dim
    def __call__(self):
        return self.c_var()
    def __repr__(self):
        # method for printing the c_variable inside Python code
        return self.c_var.__repr__()
    def declare(self):
        # returns C++ code to declare a fixed-size arry of size dim, 
        # skipping declaration if dim=0
        if self.dim>0:
            return f"{self.dtype} {self.c_var()}[{self.dim}];"
        else:
            return ""
    def assign(self, val):
        # returns C++ code string to fill all elements of a fixed size array with a single value
        # val is a c_variable representing the value.
        return f"#pragma unroll\nfor(int k=0; k<{self.dim}; k++)\n    {self()}[k] = {cast_to(self.dtype)}({val()});\n"
        
def cast_to(dtype):
    # returns C++ code string to do a cast ; e.g. "(float)" if dtype is "float" for example
    return f"({dtype})"

def VectApply(fun, out, *args):
    # returns C++ code string to apply a scalar operation to fixed-size arrays, following broadcasting rules.
    # - fun is the scalar unary function to be applied, it must accept two c_variable inputs and output a string
    # - out and args must be c_array instances
    # 
    # Example : if out.dim = 3, arg0.dim = 1, arg1.dim = 3, 
    # it will generate the following (in pseudo-code for clarity) :
    #   #pragma unroll
    #   for(int k=0; k<out.dim; k++)
    #       fun(out[k], arg0[0], arg1[k]);
    
    dims = [out.dim] + list(arg.dim for arg in args)
    
    dimloop = max(dims)
    if not set(dims) in ({dimloop}, {1, dimloop}):
        raise ValueError("incompatible dimensions in VectApply")
    incr_out = 1 if out.dim==dimloop else 0
    outk = c_variable(f"{out()}[k*{incr_out}]" , out.dtype)
    incr_args = list((1 if arg.dim==dimloop else 0) for arg in args)
    argks = list(c_variable(f"{arg()}[k*{incr}]", arg.dtype) for (arg, incr) in zip(args, incr_args))
    return f"#pragma unroll\nfor(int k=0; k<{dimloop}; k++) {{\n    {fun(outk, *argks)} }}\n"


def VectCopy(out, arg, cast=True):
    # returns a C++ code string representing a vector copy between fixed-size arrays
    # - dim is dimension of arrays
    # - out is c_variable representing the output array
    # - arg is c_variable representing the input array
    # - optional cast=True if we want to add a (type) cast operation before the copy
    cast_string = cast_to(out.dtype) if cast else ""
    return f"#pragma unroll\nfor(int k=0; k<{out.dim}; k++)\n    {out()}[k] = {cast_string}{arg()}[k];\n"



def GetDims(Vars):
    # returns the list of dim fields (dimensions) of a list of Var instances
    return tuple(v.dim for v in Vars)
    
def GetInds(Vars):
    # returns the list of ind fields (indices) of a list of Var instances
    return tuple(v.ind for v in Vars)
    


def keops_exp(x):
    # returns the C++ code string for the exponential function applied to a C++ variable
    # - x must be of type c_variable
    if x.dtype in ["float","double"]:
        return f"exp({x()})"
    else:
        raise ValueError("not implemented.")







