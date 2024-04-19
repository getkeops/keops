from c_block import c_block
from c_expression import c_expression
from c_instruction import c_instruction
from misc import Error


class c_if(c_block):

    def __init__(self, condition=c_expression(), body=c_instruction(), decorator=None):
        if not isinstance(condition, c_expression):
            Error("invalid condition")
        super().__init__(body=body, decorator=decorator, headers=(condition,))

    @property
    def pre_code_string(self):
        header = "" if self.headers == () else self.headers[0]
        return f"if({header})"
