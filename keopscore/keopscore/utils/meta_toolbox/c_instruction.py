from .c_code import c_code
from .misc import Meta_Toolbox_Error


class c_instruction(c_code):

    end_str = ";"

    def __init__(self, string, local_vars, global_vars, **kwargs):
        super().__init__(string=string, vars=global_vars.union(local_vars), **kwargs)
        self.local_vars = local_vars
        self.global_vars = global_vars

    @property
    def outer_local_vars(self):
        return self.local_vars

    def __repr__(self):
        return super().__repr__() + self.end_str

    def __add__(self, other):
        if not isinstance(other, c_instruction):
            Meta_Toolbox_Error(f"cannot add c_instruction and {type(other)}")
        if self.code_string == "":
            return other
        elif other.code_string == "":
            return self
        string = str(self) + "\n"
        string += str(other)
        locvars1, locvars2 = self.outer_local_vars, other.outer_local_vars
        if not locvars1.isdisjoint(locvars2):
            Meta_Toolbox_Error(
                f"cannot combine instructions: the following variables are defined in both parts: {locvars1.intersection(locvars2)}"
            )
        local_vars = locvars1.union(locvars2)
        global_vars = self.global_vars.union(other.global_vars).difference(local_vars)
        return c_composed_instruction(string, local_vars, global_vars)


class c_composed_instruction(c_instruction):
    end_str = ""


c_empty_instruction = c_instruction("", set(), set())
