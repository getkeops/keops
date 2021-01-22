

def value(pdtype):
    # converts string "dtype*" to "dtype" 
    if pdtype[-1]=="*":
        return pdtype[:-1]
    else:
        raise ValueError("Incorrect input string in value function; it should represent a pointer C++ type.")

def pointer(dtype):
    # converts string "dtype" to "dtype*" 
    return dtype+"*"

def declare_array(x, dim):
    # returns C++ code to declare a fixed-size arry of size dim, 
    # skipping declaration if dim=0
    if dim>0:
        return f"{value(x.dtype)} {x()}[{dim}];"
    else:
        return ""
        
def cast_to(dtype):
    # returns C++ code string to do a cast ; e.g. "(float)" if dtype is "float" for example
    return f"({dtype})"

def VectAssign(dim, out, val):
    # returns C++ code string to fill all elements of a fixed size array with a single value
    # out is c_variable representing the array, dim is the size, val is a c_variable representing the value.
    return f"#pragma unroll\nfor(int k=0; k<{dim}; k++)\n    {out()}[k] = {cast_to(value(out.dtype))}({val()});"

def VectApply(fun, dimout, dimin, out, arg):
    # returns C++ code string to apply a scalar unary operation to fixed-size arrays, following broadcasting rules.
    # Possible cases are :
    # a) if dimin=dimout, it will generate the following (in pseudo-code for clarity) :
    #   #pragma unroll
    #   for(int k=0; k<dimout; k++)
    #       fun(out[k], arg[k]);
    # b) if dimin=1, it will generate :
    #   #pragma unroll
    #   for(int k=0; k<dimout; k++)
    #       fun(out[k], arg[0]);
    # c) if dimout=1, it will generate :
    #   #pragma unroll
    #   for(int k=0; k<dimin; k++)
    #       fun(out[0], arg[k]);
    #
    # - fun is the scalar unary function to be applied, it must accpet two c_variable inputs and output a string
    # - dimout is the size of fixed-size arry represented by variable out, the ouptut
    # - dimin is the size of fixed-size arry represented by variable arg, the input
    # - arg is a c_variable representing the input.
    dimloop = max(dimout, dimin)
    if not ( (dimout==dimloop or dimout==1) and (dimin==dimloop or dimin==1) ):
        raise ValueError("incompatible dimensions in VectApply")
    incr_out = 1 if dimout==dimloop else 0
    incr_in = 1 if dimin==dimloop else 0
    outk = c_variable( f"{out()}[k*{incr_out}]" , value(out.dtype) )
    argk = c_variable( f"{arg()}[k*{incr_in}]" , value(arg.dtype) )
    return f"#pragma unroll\nfor(int k=0; k<{dimloop}; k++) {{\n    {fun(outk, argk)} }}\n"

def VectApply2(fun, dimout, dimin0, dimin1, out, arg0, arg1):
    # returns C++ code string to apply a scalar binary operation to fixed-size arrays, following broadcasting rules.
    # There are 8 possible cases, such as :
    # a) if dimin0=dimin1=dimout, it will generate the following (in pseudo-code for clarity) :
    #   #pragma unroll
    #   for(int k=0; k<dimout; k++)
    #       fun(out[k], arg0[k], arg1[k]);
    # b) if dimin0=1 and dimin1=dimout, it will generate :
    #   #pragma unroll
    #   for(int k=0; k<dimout; k++)
    #       fun(out[k], arg0[0], arg1[k]);
    # c) if dimin0=dimin1=1, it will generate :
    #   #pragma unroll
    #   for(int k=0; k<dimin; k++)
    #       fun(out[k], arg0[0], arg1[0]);
    # etc.
    #
    # - fun is the scalar unary function to be applied, it must accpet two c_variable inputs and output a string
    # - dimout is the size of fixed-size arry represented by variable out, the ouptut
    # - dimin0 is the size of fixed-size arry represented by variable arg0, the first input
    # - dimin1 is the size of fixed-size arry represented by variable arg1, the second input
    # - arg0 is a c_variable representing the first input
    # - arg1 is a c_variable representing the second input.
    dimloop = max(dimout, dimin0, dimin1)
    if not ( (dimout==dimloop or dimout==1) and (dimin0==dimloop or dimin0==1) and (dimin1==dimloop or dimin1==1) ):
        raise ValueError("incompatible dimensions in VectApply2")
    incr_out = 1 if dimout==dimloop else 0
    incr_in0 = 1 if dimin0==dimloop else 0
    incr_in1 = 1 if dimin1==dimloop else 0
    outk = c_variable( f"{out()}[k*{incr_out}]" , value(out.dtype) )
    arg0k = c_variable( f"{arg0()}[k*{incr_in0}]" , value(arg0.dtype) )
    arg1k = c_variable( f"{arg1()}[k*{incr_in1}]" , value(arg1.dtype) )
    return f"#pragma unroll\nfor(int k=0; k<{dimloop}; k++) {{\n    {fun(outk, arg0k, arg1k)} }}\n"

def VectCopy(dim, out, arg, cast=True):
    # returns a C++ code string representing a vector copy between fixed-size arrays
    # - dim is dimension of arrays
    # - out is c_variable representing the output array
    # - arg is c_variable representing the input array
    # - optional cast=True if we want to add a (type) cast operation before the copy
    if cast:
        return f"#pragma unroll\nfor(int k=0; k<{dim}; k++)\n    {out()}[k] = {cast_to(value(out.dtype))}{arg()}[k];\n"
    else:
        return f"#pragma unroll\nfor(int k=0; k<{dim}; k++)\n    {out()}[k] = {arg()}[k];\n"

class new_c_variable:
    # class to generate instances of c_variable with unique names, to avoid conflicts in C++ code
    dict_instances = {}
    def __new__(self, template_string_id, dtype):
        # - template_string_id is a string, the base name for c_variable
        # - dtype is a string, the C++ type of the c_variable to create
        # For example the first call to new_c_variable("x","float")
        # will create a c_variable with string_id="x_1", the second call 
        # will create a c_variable with string_id="x_2", etc.        
        if template_string_id in new_c_variable.dict_instances:
            cnt = new_c_variable.dict_instances[template_string_id] + 1
        else:
            cnt = 1
        new_c_variable.dict_instances[template_string_id] = cnt
        string_id = template_string_id + "_" + str(cnt)
        return c_variable(string_id,dtype)
            
class c_variable:
    # class to represent a C++ variable, storing its c++ name and its C++ type.
    def __init__(self, string_id, dtype):
        self.string_id = string_id      # string_id is C++ name of variable
        self.dtype = dtype              # dtype is C++ type of variable
    def __call__(self):
        # ouputs string_id, the C++ name of the variable
        return self.string_id
    def __repr__(self):
        # method for printing the c_variable inside Python code
        return self.string_id

def GetDims(Vars):
    # returns the list of dim fields (dimensions) of a list of Var instances
    return tuple(v.dim for v in Vars)
    
def GetInds(Vars):
    # returns the list of ind fields (indices) of a list of Var instances
    return tuple(v.ind for v in Vars)
    
    
def load_vars(dims, inds, i, xi, px, table=None):
    # returns a c++ code used to create a local copy of slices of the input tensors, for evaluating a formula
    # - dims is a list of integers specifying "dimension" of each input tensor ; each tensor being interpreted as a matrix of shape (n,dim)
    # - inds is a list of integers specifying "indices" of each input tensor, i.e. its position in the input px (which is a pointer to pointer)
    # - i is a c_variable (of dtype="int"), specifying which row of the matrix should be loaded
    # - xi is a c_variable, representing the local copy of the ith rows of all input tensors
    # - px is a c_variable, representing a pointer to all input tensors 
    #
    # Example: assuming i=c_variable("5","int"), xi=c_variable("xi","float") and px=c_variable("px","float**"), then 
    #   load_vars ( [2,2,3], [7,9,8], i, xi, px )
    # will output the following code:
    #   xi[0] = px[7][5*2+0];
    #   xi[1] = px[7][5*2+1];
    #   xi[3] = px[9][5*2+0];
    #   xi[4] = px[9][5*2+1];
    #   xi[5] = px[8][5*3+0];
    #   xi[6] = px[8][5*3+1];
    #   xi[7] = px[8][5*3+2];
    #
    # N.B. the optional table argument is used to build the table of c_variables, used in the Eval methods of operations.
    
    string = ""
    k = 0
    for u in range(len(dims)):
        if table is not None:
            table[inds[u]] = c_variable(f"({xi()}+{k})",xi.dtype)
        for v in range(dims[u]):
            string += f"{xi()}[{k}] = {px()}[{inds[u]}][{i()}*{dims[u]}+{v}];\n"
            k+=1
    if table is not None:
        return string, table
    else:
        return string
    
    


def keops_exp(x):
    # returns the C++ code string for the exponential function applied to a C++ variable
    # - x must be of type c_variable
    if x.dtype in ["float","double"]:
        return f"exp({x()})"







