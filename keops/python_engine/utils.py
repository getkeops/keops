from hashlib import sha256
 
def get_hash_name(*args):
    return sha256("".join(list(str(arg) for arg in args)).encode("utf-8")).hexdigest()[:10]
    
def c_if(condition, *commands):
    block_string = "".join(commands)
    return f""" if ({cond_string}) {{
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
    def declare(self):
        return f"{self.dtype} {self.string_id}\n"
    def declare_assign(self, value_string):
        return f"{self.dtype} {self.string_id} = {value_string}\n"
        
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
    



class Var_loader:
    
    def __init__(self, red_formula):
        
        formula = red_formula.formula
        tagI, tagJ = red_formula.tagI, red_formula.tagJ
        
        self.Varsi = formula.Vars(cat=tagI)         # list all "i"-indexed variables in the formula
        self.indsi = GetInds(self.Varsi)            # list indices of "i"-indexed variables
        self.dimsx = GetDims(self.Varsi)            # list dimensions of "i"-indexed variables
        self.dimx = sum(self.dimsx)                 # total dimension of "i"-indexed variables
        
        self.Varsj = formula.Vars(cat=tagJ)         # list all "j"-indexed variables in the formula
        self.indsj = GetInds(self.Varsj)            # list indices of "j"-indexed variables
        self.dimsy = GetDims(self.Varsj)            # list dimensions of "j"-indexed variables
        self.dimy = sum(self.dimsy)                 # total dimension of "j"-indexed variables
        
        self.Varsp = formula.Vars(cat=2)            # list all parameter variables in the formula
        self.indsp = GetInds(self.Varsp)            # list indices of parameter variables
        self.dimsp = GetDims(self.Varsp)            # list indices of parameter variables
        self.dimp = sum(self.dimsp)                 # total dimension of parameter variables
        
        self.inds = GetInds(formula.Vars_)
        self.nminargs = max(self.inds)+1 if len(self.inds)>0 else 0

    def table(self, xi, yj, pp):    
        table = [None] * self.nminargs
        for (dims, inds, xloc) in ((self.dimsx, self.indsi, xi), (self.dimsy, self.indsj, yj), (self.dimsp, self.indsp, pp)):
            k = 0
            for u in range(len(dims)):
                table[inds[u]] = c_array(f"({xloc()}+{k})", xloc.dtype, dims[u])
                k += dims[u]
        return table
    
    def direct_table(self, args, i, j):    
        table = [None] * self.nminargs
        for (dims, inds, row_index) in ((self.dimsx, self.indsi, i), (self.dimsy, self.indsj, j), (self.dimsp, self.indsp, c_zero_int)):
            for u in range(len(dims)):
                arg = args[inds[u]]
                table[inds[u]] = c_array(f"({arg()}+{row_index()}*{dims[u]})", value(arg.dtype), dims[u])
        return table
    
    def load_vars(self, cat, xloc, args, row_index=c_zero_int):
        # returns a c++ code used to create a local copy of slices of the input tensors, for evaluating a formula
        # cat is either "i", "j" or "p", specifying the category of variables to be loaded
        # - xloc is a c_array, the local array which will receive the copy
        # - args is a list of c_variable, representing pointers to input tensors 
        # - row_index is a c_variable (of dtype="int"), specifying which row of the matrix should be loaded
        #
        # Example: assuming i=c_variable("5","int"), xloc=c_variable("xi","float") and px=c_variable("px","float**"), then 
        # if self.dimsx = [2,2,3] and self.indsi = [7,9,8], the call to
        #   load_vars ( "i", xi, [arg0, arg1,..., arg9], row_index=i )
        # will output the following code:
        #   xi[0] = arg7[5*2+0];
        #   xi[1] = arg7[5*2+1];
        #   xi[3] = arg9[5*2+0];
        #   xi[4] = arg9[5*2+1];
        #   xi[5] = arg8[5*3+0];
        #   xi[6] = arg8[5*3+1];
        #   xi[7] = arg8[5*3+2];
        
        if cat=="i":
            dims, inds  = self.dimsx, self.indsi
        elif cat=="j":
            dims, inds = self.dimsy, self.indsj
        elif cat=="p":
            dims, inds = self.dimsp, self.indsp
        string = ""
        k = 0
        for u in range(len(dims)):
            for v in range(dims[u]):
                string += f"{xloc()}[{k}] = {args[inds[u]]()}[{row_index()}*{dims[u]}+{v}];\n"
                k+=1
        return string   
        
        
        
        
        
def keops_exp(x):
    # returns the C++ code string for the exponential function applied to a C++ variable
    # - x must be of type c_variable
    if x.dtype in ["float","double"]:
        return f"exp({x()})"
    else:
        raise ValueError("not implemented.")







