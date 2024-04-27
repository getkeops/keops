from .c_block import c_block
from .c_expression import c_expression, c_empty_expression
from .c_instruction import c_empty_instruction, c_composed_instruction
from .misc import Meta_Toolbox_Error


class c_if(c_block):

    def __init__(
        self, condition=c_empty_expression, body=c_empty_instruction, **kwargs
    ):
        if not isinstance(condition, c_expression):
            Meta_Toolbox_Error("invalid condition")
        if isinstance(body, tuple):
            body = sum(body, start=c_empty_instruction)
        headers = (condition,) if body != c_empty_instruction else ()
        use_braces = isinstance(body, c_composed_instruction) or isinstance(body, tuple)
        super().__init__(body=body, headers=headers, use_braces=use_braces, **kwargs)

    @property
    def pre_code_string(self):
        header_string = (
            "" if self.headers == () else self.headers[0].code_string_no_parenthesis
        )
        return f"if({header_string})"
