from keopscore.utils.meta_toolbox.c_variable import c_variable
from .c_instruction import c_empty_instruction
from .c_block import c_block
from .c_code import c_code
from .c_expression import c_expression
from .misc import Meta_Toolbox_Error, new_c_name


class c_function(c_block):

    def __init__(self, dtype_out="void", name=None, input_vars=(), body=c_empty_instruction, **kwargs):
        if name is None:
            name = new_c_name("fun")
        self.name = name
        self.dtype_out = dtype_out
        self.input_vars = input_vars
        headers = tuple(var.declare() for var in input_vars)
        super().__init__(body=body, headers=headers, **kwargs)

    @property
    def pre_code_string(self):
        return f"{self.dtype_out} {self.name}({', '.join(x.code_string for x in self.headers)})"

    def __call__(self, *vars):
        vars_dtype = [var.dtype for var in vars]
        input_vars_dtype = [var.dtype for var in self.input_vars]
        if vars_dtype != input_vars_dtype:
            Meta_Toolbox_Error(
                f"invalid inputs for call to c_function {self.name}. Signature should be {input_vars_dtype} but received {vars_dtype}"
            )
        str_args = ", ".join(str(x) for x in vars)
        expression = f"{self.name}({str_args})"
        return c_expression(expression, vars, self.dtype_out, add_parenthesis=False)



class cuda_global_kernel(c_function):

    blockIdx_x = c_variable("int", "blockIdx.x")
    blockDim_x = c_variable("int", "blockDim.x")
    threadIdx_x = c_variable("int", "threadIdx.x")

    def __init__(self, name, input_vars=(), body=c_empty_instruction, **kwargs):
        super().__init__('extern "C" __global__ void', name, input_vars, body, **kwargs)
