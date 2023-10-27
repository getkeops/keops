from keopscore.utils.code_gen_utils import (
    c_for_loop,
    new_c_varname,
    c_variable,
)
from keopscore.utils.misc_utils import KeOps_Error

import keopscore.config.config


def math_function(
    cpu_code, gpu_code=None, gpu_half2_code=None, gpu_float_code=None, void=False
):
    if gpu_code is None:
        gpu_code = cpu_code
    if gpu_half2_code is None:
        gpu_half2_code = gpu_code
    if gpu_float_code is None:
        gpu_float_code = gpu_code

    def convert_to_fun(code):
        if isinstance(code, str):
            code_fun = lambda *args: code + "(" + ",".join(arg for arg in args) + ")"
        else:
            code_fun = code
        return code_fun

    def call(*args):
        args = list(args)
        for k, arg in enumerate(args):
            if isinstance(arg, int):
                c_dtype = "signed long int" if arg > 2e9 else "int"
                args[k] = c_variable(c_dtype, str(arg))
        # N.B. first argument gives main dtype
        dtype = args[0].dtype
        if dtype == "half2":
            if gpu_half2_code == "NA":
                KeOps_Error("Operation is not implemented for half precision")
            code_fun_gpu = convert_to_fun(gpu_half2_code)
        elif dtype == "float":
            code_fun_gpu = convert_to_fun(gpu_float_code)
        else:
            code_fun_gpu = convert_to_fun(gpu_code)
        string_gpu = code_fun_gpu(*(arg.id for arg in args))
        code_fun_cpu = convert_to_fun(cpu_code)
        string_cpu = code_fun_cpu(*(arg.id for arg in args))
        string = f"""
                    #ifdef __CUDACC__
                        {string_gpu}
                    #else
                        {string_cpu}
                    #endif
                """
        if void:
            return string
        else:
            return c_variable(dtype, string)

    return call


# helpers for half2
h2zero = "__float2half2_rn(0.0f)"
h2one = "__float2half2_rn(1.0f)"
h2ge0 = lambda x: f"__hge2({x},{h2zero})"
h2le0 = lambda x: f"__hle2({x},{h2zero})"
h2eq0 = lambda x: f"__heq2({x},{h2zero})"
h2ifelse = lambda x, a, b: f"(({b})+(({a})-({b}))*{h2ge0(x)})"
int2h2 = lambda x: f"__float2half2_rn((float){x})"


keops_mul = math_function(cpu_code=lambda x, y: f"({x}*{y})", gpu_half2_code="__hmul2")

keops_abs = math_function(cpu_code="abs", gpu_half2_code="__habs2")
keops_cos = math_function(cpu_code="cos", gpu_half2_code="h2cos")
keops_sin = math_function(cpu_code="sin", gpu_half2_code="h2sin")
keops_sinxdivx = math_function(
    cpu_code=lambda x: f"({x} ? sin({x})/{x} : 1.0f)",
    gpu_half2_code=lambda x: f"({h2one}*{h2eq0(x)}+h2sin({x})/({x}+{h2eq0(x)}))",
)
keops_acos = math_function(cpu_code="acos", gpu_half2_code="NA")
keops_asin = math_function(cpu_code="asin", gpu_half2_code="NA")
keops_atan = math_function(cpu_code="atan", gpu_half2_code="NA")
keops_atan2 = math_function(cpu_code="atan2", gpu_half2_code="NA")
keops_exp = math_function(cpu_code="exp", gpu_half2_code="h2exp")
keops_floor = math_function(cpu_code="floor", gpu_half2_code="h2floor")
keops_log = math_function(cpu_code="log", gpu_half2_code="h2log")
keops_xlogx = math_function(
    cpu_code=lambda x: f"({x} ? {x} * log({x}) : 0.0f)",
    gpu_half2_code=lambda x: f"(h2log({x}+{h2eq0(x)})*({x}))",
)
keops_fma = math_function(cpu_code="fma", gpu_half2_code="__hfma2")
keops_pow = math_function(
    cpu_code="pow",
    gpu_code="powf",
    gpu_half2_code=lambda x, y: f"(h2exp({int2h2(y)}*h2log({x})))",
)
keops_powf = math_function(
    cpu_code="powf", gpu_half2_code=lambda x, y: f"h2exp({y}*h2log({x}))"
)
keops_rcp = math_function(cpu_code=lambda x: f"(1.0f/({x}))", gpu_half2_code="h2rcp")

keops_rsqrt = math_function(
    cpu_code=lambda x: f"(({x}==0.0f)? 0.0f : 1.0f/sqrt({x}))",
    gpu_code=lambda x: f"(({x}==0.0f)? 0.0f : rsqrt({x}))",
    gpu_half2_code=lambda x: f"(h2rsqrt({x}+{h2eq0(x)}) * ({h2one}-{h2eq0(x)}))",
)

keops_sqrt = math_function(cpu_code="sqrt", gpu_half2_code="h2sqrt")

keops_relu = math_function(
    cpu_code=lambda x: f"(({x}<0.0f)? 0.0f : {x})",
    gpu_half2_code=lambda x: f"{h2ge0(x)}*({x})",
)

keops_equal = math_function(
    cpu_code=lambda x, y: f"(({x}=={y})? 1.0f : 0.0f)", gpu_half2_code="__heq2"
)

keops_notequal = math_function(
    cpu_code=lambda x, y: f"(({x}!={y})? 1.0f : 0.0f)", gpu_half2_code="__hne2"
)

keops_lessthan = math_function(
    cpu_code=lambda x, y: f"(({x}<{y})? 1.0f : 0.0f)", gpu_half2_code="__hlt2"
)

keops_lessorequal = math_function(
    cpu_code=lambda x, y: f"(({x}<={y})? 1.0f : 0.0f)", gpu_half2_code="__hle2"
)

keops_step = math_function(
    cpu_code=lambda x: f"(({x}<0.0f)? 0.0f : 1.0f)", gpu_half2_code=h2ge0
)

keops_sign = math_function(
    cpu_code=lambda x: f"(({x}>0.0f)? 1.0f : ( ({x}<0.0f)? -1.0f : 0.0f ))",
    gpu_half2_code=lambda x: f"{h2ge0(x)}-{h2le0(x)}",
)


keops_ifelse = math_function(
    cpu_code=lambda x, a, b: f"(({x}>=0.0f) ? {a} : {b})", gpu_half2_code=h2ifelse
)

keops_clamp = math_function(
    cpu_code=lambda x, a, b: f"(({x}<{a})? {a} : ( ({x}>{b})? {b} : {x} ))",
    gpu_half2_code=lambda x, a, b: h2ifelse(f"{x}-{a}", h2ifelse(f"{b}-{x}", x, b), a),
)
keops_clampint = math_function(
    cpu_code=lambda x, a, b: f"(({x}<{a})? {a} : ( ({x}>{b})? {b} : {x} ))",
    gpu_half2_code=lambda x, a, b: h2ifelse(
        f"{x}-{int2h2(a)}", h2ifelse(f"{int2h2(b)}-{x}", x, int2h2(b)), int2h2(a)
    ),
)
keops_mod = math_function(
    cpu_code=lambda x, n, d: f"({x} - {n} * floor(({x} - {d})/{n}))",
    gpu_half2_code="NA",
)
keops_round = math_function(
    cpu_code=lambda x, d: f"round({x})"
    if eval(d) == 0
    else f"(round({x}*{10**eval(d)})/{10**eval(d)})",
    gpu_half2_code="NA",
)
keops_diffclampint = math_function(
    cpu_code=lambda x, a, b: f"(({x}<{a})? 0.0f : ( ({x}>{b})? 0.0f : 1.0f ))",
    gpu_half2_code=lambda x, a, b: h2ifelse(
        f"{x}-{a}", h2ifelse(f"{b}-{x}", h2one, h2zero), h2zero
    ),
)

keops_sincos = math_function(
    cpu_code=lambda x, s, c: f"*({s})=sin({x}); *({c})=cos({x});",
    gpu_half2_code="NA",
    void=True,
)
