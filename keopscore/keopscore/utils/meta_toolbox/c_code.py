from .misc import Meta_Toolbox_Error


class c_code:

    def __init__(self, string="", vars=set(), comment=None):
        if not isinstance(string, str):
            Meta_Toolbox_Error("should be string")
        if comment is not None:
            string = "// " + comment + "\n" + string
        self.code_string = string
        self.vars = vars

    def __repr__(self):
        return self.code_string

    def __str__(self):
        return self.__repr__()

    def __add__(self, other):
        if not isinstance(other, c_code):
            Meta_Toolbox_Error("other should be c_code instance")
        return c_code(
            self.code_string + other.code_string, vars=self.vars.union(other.vars)
        )


def c_include(*headers, **kwargs):
    return c_code("".join(f"#include <{header}>\n" for header in headers), **kwargs)


def c_define(name, value, **kwargs):
    return c_code(f"#define {name} {value}\n", **kwargs)


c_line_break = c_code("\n")
