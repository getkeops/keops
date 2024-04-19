def Error(message):
    raise ValueError(message)


global_indent = "    "

registered_dtypes = (
    "void",
    "float",
    "double",
    "half",
    "int",
    "signed long int",
    "float2",
    "bool",
)
registered_dtypes = registered_dtypes + tuple(x + "*" for x in registered_dtypes)

disable_pragma_unrolls = False


def use_pragma_unroll(n=64):
    if disable_pragma_unrolls:
        return "\n"
    else:
        if n is None:
            return f"\n#pragma unroll\n"
        else:
            return f"\n#pragma unroll({n})\n"


def sizeof(dtype):
    if dtype in ("float", "int"):
        return 4
    elif dtype == ("double", "signed long int", "float2"):
        return 8
    elif dtype == "half":
        return 2
    else:
        Error("not implemented")


def to_tuple(x):
    if not hasattr(x, "__iter__"):
        return (x,)
    else:
        return tuple(x)


def add_indent(block_str):
    string = ""
    lines = block_str.split("\n")
    for line in lines:
        if line == "":
            pass
        else:
            string += global_indent + line + "\n"
    return string


class new_c_name:
    # class to generate unique names for variables in C++ code, to avoid conflicts
    dict_instances = {}

    def __new__(self, template_string_id, num=1, as_list=False):
        # - template_string_id is a string, the base name for c_variable
        # - if num>1 returns a list of num new names with same base names
        # For example the first call to new_c_variable("x")
        # will return "x_1", the second call will return "x_2", etc.
        if num > 1 or as_list:
            return list(new_c_name(template_string_id) for k in range(num))
        if template_string_id in new_c_name.dict_instances:
            cnt = new_c_name.dict_instances[template_string_id] + 1
        else:
            cnt = 0
        new_c_name.dict_instances[template_string_id] = cnt
        string_id = template_string_id + "_" + str(cnt)
        return string_id


def call_list(args):
    return ", ".join(list(arg.id for arg in args))


def signature_list(args):
    return ", ".join(list(f"{arg.dtype} {arg.id}" for arg in args))


def c_include(*headers):
    return "".join(f"#include <{header}>\n" for header in headers)
