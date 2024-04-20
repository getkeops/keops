from .c_instruction import c_instruction
from .c_expression import c_expression, py2c, cast_to


class c_lvalue(c_expression):
    # class to represent a C++ l-value

    def __init__(self, dtype, vars, string_id, add_parenthesis):
        super().__init__(
            string_id, vars=vars, dtype=dtype, add_parenthesis=add_parenthesis
        )
        self.id = string_id

    def assign(self, value, assign_op="="):
        value = py2c(value)
        local_vars = set()
        global_vars = self.vars.union(value.vars)
        if value.dtype != self.dtype:
            if self.dtype == "float2" and value.dtype == "float":
                assign_x = c_instruction(
                    f"{self}.x {assign_op} {value.code_string}", local_vars, global_vars
                )
                assign_y = c_instruction(
                    f"{self}.y {assign_op} {value.code_string}", local_vars, global_vars
                )
                return assign_x + assign_y
            else:
                return c_instruction(
                    f"{self} {assign_op} {value}",
                    local_vars,
                    global_vars,
                )
        else:
            return c_instruction(
                f"{self} {assign_op} {value.code_string}", local_vars, global_vars
            )

    def add_assign(self, value):
        return self.assign(value, assign_op="+=")

    @property
    def plus_plus(self):
        return c_instruction(f"{self}++", set(), self.vars)