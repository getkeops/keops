from .c_lvalue import c_lvalue
from .c_instruction import c_instruction
from .c_expression import py2c, cast_to
from .misc import Meta_Toolbox_Error, new_c_name


class c_variable(c_lvalue):
    # class to represent a C++ variable, storing its c++ name and its C++ type.
    def __new__(self, dtype, string_id=None, qualifier=None):
        if isinstance(string_id, list):
            return list(
                c_variable(dtype, string_id, qualifier) for string_id in string_id
            )
        else:
            return super(c_variable, self).__new__(self)

    def __init__(self, dtype, string_id=None, qualifier=None):
        if string_id is None:
            string_id = new_c_name("var")
        super().__init__(
            string_id=string_id, vars=set([self]), dtype=dtype, add_parenthesis=False
        )
        if qualifier != None:
            self.declaration_string = qualifier + " " + dtype
        else:
            self.declaration_string = dtype

    def declare(self):
        local_vars = self.vars
        global_vars = set()
        return c_instruction(
            f"{self.declaration_string} {self}", local_vars, global_vars
        )

    def declare_assign(self, value, **kwargs):
        value = py2c(value)
        local_vars = self.vars
        global_vars = value.vars
        return c_instruction(
            f"{self.declaration_string} {self.assign(value).code_string}",
            local_vars,
            global_vars,
            **kwargs,
        )
