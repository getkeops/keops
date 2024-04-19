from .misc import Meta_Toolbox_Error


class c_code:

    def __init__(self, string="", vars=set()):
        if not isinstance(string, str):
            Meta_Toolbox_Error("should be string")
        self.code_string = string
        self.vars = vars

    def __repr__(self):
        return self.code_string

    def __str__(self):
        return self.__repr__()
