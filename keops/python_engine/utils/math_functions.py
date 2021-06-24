from keops.python_engine.utils.code_gen_utils import c_for_loop, new_c_varname, c_variable
from keops.python_engine import use_cuda

def math_function(cpu_code, gpu_code=None, gpu_half2_code=None, void=False):
    if gpu_code is None:
        gpu_code = cpu_code
    if gpu_half2_code is None:
        gpu_half2_code = cpu_code
    def convert_to_fun(code):
        if isinstance(code, str):
            code_fun = lambda *args : code + "(" + ",".join(arg for arg in args) + ")"
        else:
            code_fun = code
        return code_fun
    def call(*args):
        args = list(args)
        for k, arg in enumerate(args):
            if isinstance(arg, int):
                args[k] = c_variable("int", str(arg))
        # N.B. first argument gives main dtype
        dtype = args[0].dtype
        if dtype == "half2":
            code_fun = convert_to_fun(gpu_half2_code)
        elif use_cuda:
            code_fun = convert_to_fun(gpu_code)
        else:
            code_fun = convert_to_fun(cpu_code)
        string = code_fun(*(arg.id for arg in args))
        if void:
            return string
        else:
            return c_variable(dtype, string)
    return call

keops_abs = math_function(cpu_code = "abs")
keops_cos = math_function(cpu_code = "cos")
keops_sin = math_function(cpu_code = "sin")
keops_sinxdivx = math_function(cpu_code = lambda x : f"({x} ? sin({x})/{x} : 1.0f)") 
keops_acos = math_function(cpu_code = "acos")
keops_asin = math_function(cpu_code = "asin")
keops_atan = math_function(cpu_code = "atan")
keops_atan2 = math_function(cpu_code = "atan2")
keops_exp = math_function(cpu_code = "exp")
keops_floor = math_function(cpu_code = "floor")
keops_log = math_function(cpu_code = "log")
keops_xlogx = math_function(cpu_code = lambda x : f"({x} ? {x} * log({x}) : 0.0f)")
keops_fma = math_function(cpu_code = "fma")
keops_pow = math_function(cpu_code = "pow")
keops_powf = math_function(cpu_code = "powf")
keops_rcp = math_function(cpu_code = lambda x : f"(1.0f/({x}))")
keops_rsqrt = math_function(cpu_code = lambda x : f"(1.0f/sqrt({x}))")
keops_sqrt = math_function(cpu_code = "sqrt")

keops_relu = math_function(cpu_code = lambda x : f"(({x}<0.0f)? 0.0f : {x})") 
keops_step = math_function(cpu_code = lambda x : f"(({x}<0.0f)? 0.0f : 1.0f)") 
keops_sign = math_function(cpu_code = lambda x : f"(({x}>0.0f)? 1.0f : ( ({x}<0.0f)? -1.0f : 0.0f ))") 
keops_clamp = math_function(cpu_code = lambda x, a, b : f"(({x}<{a})? {a} : ( ({x}>{b})? {b} : {x} ))") 
keops_clampint = math_function(cpu_code = lambda x, a, b : f"(({x}<{a})? {a} : ( ({x}>{b})? {b} : {x} ))") 
keops_mod = math_function(cpu_code = lambda x, n, d : f"({x} - {n} * floor(({x} - {d})/{n}))") 
keops_round = math_function(cpu_code = lambda x, d : f"({d} ? round({x} * pow(10, {d}))/pow(10, {d}) : round({x}))")
keops_diffclampint = math_function(cpu_code = lambda x, a, b : f"(({x}<{a})? 0.0f : ( ({x}>{b})? 0.0f : 1.0f ))")
keops_ifelse = math_function(cpu_code = lambda x, a, b : f"(({x}>=0.0f) ? {a} : {b})") 

keops_sincos = math_function(cpu_code = lambda x, s, c : f"*{s}=sin(x); *{c}=cos(x);", void=True) 

