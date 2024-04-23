from .c_code import c_code
from .c_expression import c_expression
from .c_instruction import c_instruction, c_empty_instruction, c_composed_instruction
from .c_variable import c_variable
from .misc import Meta_Toolbox_Error, add_indent


class c_block(c_composed_instruction):

    def __init__(
        self,
        body=c_empty_instruction,
        headers=(),
        decorator=None,
        comment=None,
        use_braces=True,
    ):
        super().__init__("", set(), set())
        self.decorator = decorator
        self.headers = headers
        if isinstance(body, tuple):
            res = c_empty_instruction
            for elem in body:
                res += elem
            body = res
        if not isinstance(body, c_instruction):
            Meta_Toolbox_Error(
                "invalid argument to set_body; should be a c_instruction"
            )
        if body.code_string == "" and headers == ():
            return

        self.body = body

        self.code_string = ""
        if comment is not None:
            self.code_string += "\n// " + comment + "\n"
        if self.decorator is not None:
            self.code_string += str(self.decorator) + "\n"
        if self.pre_code_string is not None:
            self.code_string += str(self.pre_code_string) + "\n"
        if use_braces:
            self.code_string += "{\n"
        self.code_string += add_indent(str(self.body))
        if use_braces:
            self.code_string += "\n}"

        tmp = c_empty_instruction
        for code in (*self.headers, self.body):
            if isinstance(code, c_instruction):
                tmp += code
            elif isinstance(code, c_expression):
                for var in code.vars:
                    if var not in self.vars:
                        tmp.global_vars.add(var)
        self.local_vars = tmp.local_vars
        self.global_vars = tmp.global_vars

    @property
    def outer_local_vars(self):
        return set()

    @property
    def pre_code_string(self):
        return None
