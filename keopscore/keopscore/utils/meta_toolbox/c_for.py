from .c_block import c_block
from .c_code import c_code
from .c_expression import c_expression, py2c
from .c_instruction import c_instruction, c_empty_instruction
from .c_variable import c_variable
from .misc import Meta_Toolbox_Error, new_c_name, to_tuple, use_pragma_unroll


class c_for(c_block):

    def __init__(
        self,
        init=(),
        end=c_empty_instruction,
        loop=(),
        body=c_empty_instruction,
        decorator=None,
    ):
        init_instructions = to_tuple(init)
        loop_instructions = to_tuple(loop)
        end_expression = end
        if not all(
            isinstance(x, c_instruction) for x in init_instructions + loop_instructions
        ) or not isinstance(end_expression, c_expression):
            Meta_Toolbox_Error("invalid arguments")
        self.init_instructions = init_instructions
        self.end_expression = end_expression
        self.loop_instructions = loop_instructions
        headers = (
            *init_instructions,
            end_expression,
            *loop_instructions,
        )
        super().__init__(body=body, decorator=decorator, headers=headers)

    @property
    def pre_code_string(self):
        strings = []
        strings = (
            ",".join(x.code_string for x in self.init_instructions),
            self.end_expression.code_string,
            ",".join(x.code_string for x in self.loop_instructions),
        )
        header = "; ".join(string for string in strings)
        return f"for({header})"

def c_for_loop(start, end, incr, pragma_unroll=False, name_incr=None):

    start, end, incr = map(py2c, (start, end, incr))

    if pragma_unroll:
        decorator = use_pragma_unroll()
    else:
        decorator = None

    if any(x.dtype for x in (start, end, incr)) == "signed long int":
        type_incr = "signed long int"
    else:
        type_incr = "int"

    k = c_variable(type_incr, string_id=name_incr)

    def printfun(body):
        if isinstance(body, str):
            Meta_Toolbox_Error("should not be str")
        for_loop = c_for(
            init=k.declare_assign(start),
            end=k < end,
            loop=k.add_assign(incr),
            body=body,
            decorator=decorator,
        )
        return for_loop

    return printfun, k
