def value(dtype):
    return dtype[:-1]

def pointer(dtype):
    return dtype+"*"

def declare_array(x, dim):
    if dim>0:
        return f"{value(x.dtype)} {x()}[{dim}];"
    else:
        return ""
        
def cast_to(dtype):
    return f"({dtype})"

def VectAssign(out, dim, val):
    return f"#pragma unroll\nfor(int k=0; k<{dim}; k++)\n    {out()}[k] = {cast_to(value(out.dtype))}({val()});"

def VectApply(fun, dimout, dimin, out, arg):
    dimloop = max(dimout, dimin)
    if not ( (dimout==dimloop or dimout==1) and (dimin==dimloop or dimin==1) ):
        raise ValueError("incompatible dimensions in VectApply")
    incr_out = 1 if dimout==dimloop else 0
    incr_in = 1 if dimin==dimloop else 0
    outk = c_variable( f"{out()}[k*{incr_out}]" , value(out.dtype) )
    argk = c_variable( f"{arg()}[k*{incr_in}]" , value(arg.dtype) )
    return f"#pragma unroll\nfor(int k=0; k<{dimloop}; k++) {{\n    {fun(outk, argk)} }}\n"

def VectApply2(fun, dimout, dimin0, dimin1, out, arg0, arg1):
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
    if cast:
        return f"#pragma unroll\nfor(int k=0; k<{dim}; k++)\n    {out()}[k] = {cast_to(value(out.dtype))}{arg()}[k];\n"
    else:
        return f"#pragma unroll\nfor(int k=0; k<{dim}; k++)\n    {out()}[k] = {arg()}[k];\n"

class new_c_variable:
    dict_instances = {}
    def __new__(self, template_string_id, dtype):
        if template_string_id in new_c_variable.dict_instances:
            cnt = new_c_variable.dict_instances[template_string_id] + 1
        else:
            cnt = 1
        new_c_variable.dict_instances[template_string_id] = cnt
        string_id = template_string_id + "_" + str(cnt)
        return c_variable(string_id,dtype)
            
class c_variable:
    def __init__(self, string_id, dtype):
        self.string_id = string_id
        self.dtype = dtype
    def __call__(self):
        return self.string_id
    def __repr__(self):
        return self.string_id

def GetDims(Vars):
    return tuple(v.dim for v in Vars)
    
def GetInds(Vars):
    return tuple(v.ind for v in Vars)
    
    
def keops_exp(x):
    if x.dtype in ["float","double"]:
        return f"exp({x()})"
    
def load_vars(dims, inds, i, xi, px, table=None):
    string = ""
    if table is None:
        table = [None]*(max(inds)+1)
    k = 0
    for u in range(len(dims)):
        table[inds[u]] = c_variable(f"({xi()}+{k})","float")
        for v in range(dims[u]):
            string += f"{xi()}[{k}] = {px()}[{inds[u]}][{i()}*{dims[u]}+{v}];\n"
            k+=1
    return string, table
    