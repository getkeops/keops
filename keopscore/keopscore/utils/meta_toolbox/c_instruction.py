from c_code import c_code
from misc import Error


class c_instruction(c_code):

    def __init__(self, string="", local_vars=set(), global_vars=set(), end_str=";\n"):
        super().__init__(string=string, vars=global_vars.union(local_vars))
        self.local_vars = local_vars
        self.global_vars = global_vars
        self.end_str = end_str

    def __repr__(self):
        return self.code_string + self.end_str

    @property
    def outer_local_vars(self):
        return self.local_vars

    def __add__(self, other):
        if not isinstance(other, c_instruction):
            Error(f"cannot add c_instruction and {type(other)}")
        string = str(self) + other.code_string
        locvars1, locvars2 = self.outer_local_vars, other.outer_local_vars
        if not locvars1.isdisjoint(locvars2):
            Error(
                f"cannot combine instructions: the following variables are defined in both parts: {locvars1.intersection(locvars2)}"
            )
        local_vars = locvars1.union(locvars2)
        global_vars = self.global_vars.union(other.global_vars).difference(local_vars)
        return c_instruction(string, local_vars, global_vars, end_str=other.end_str)
