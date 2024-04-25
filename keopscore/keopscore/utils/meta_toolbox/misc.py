def Meta_Toolbox_Error(message):
    raise ValueError(message)


global_indent = "    "

registered_dtypes = (
    "void",
    "float",
    "double",
    "half",
    "half2",
    "int",
    "signed long int",
    "float2",
    "bool",
)
registered_dtypes = (
    registered_dtypes
    + tuple(x + "*" for x in registered_dtypes)
    + tuple(x + "**" for x in registered_dtypes)
)


def is_pointer_dtype(dtype):
    return dtype[-1] == "*"


disable_pragma_unrolls = False


def use_pragma_unroll(n=64):
    if disable_pragma_unrolls:
        return ""
    else:
        if n is None:
            return "#pragma unroll"
        else:
            return f"#pragma unroll({n})"


def sizeof(dtype):
    if dtype in ("float", "int"):
        return 4
    elif dtype in ("double", "signed long int", "float2"):
        return 8
    elif dtype == "half":
        return 2
    else:
        Meta_Toolbox_Error("not implemented")


def to_tuple(x):
    if not hasattr(x, "__iter__"):
        return (x,)
    else:
        return tuple(x)


def add_indent(block_str):
    lines = block_str.split("\n")
    indented_lines = [global_indent + line for line in lines]
    return "\n".join(indented_lines)


class new_c_name:
    # class to generate unique names for variables in C++ code, to avoid conflicts
    dict_instances = {}

    def __new__(self, template_string_id, num=1, as_list=False):
        # - template_string_id is a string, the base name for c_variable
        # - if num>1 returns a list of num new names with same base names
        # For example the first call to new_c_variable("x")
        # will return "x", the second call will return "x2", etc.
        if num > 1 or as_list:
            return list(new_c_name(template_string_id) for k in range(num))
        if template_string_id in new_c_name.dict_instances:
            cnt = new_c_name.dict_instances[template_string_id] + 1
        else:
            cnt = 1
        new_c_name.dict_instances[template_string_id] = cnt
        string_id = template_string_id
        if cnt > 1:
            string_id += str(cnt)
        return string_id


def call_list(args):
    return ", ".join(list(arg.id for arg in args))


def signature_list(args):
    return ", ".join(list(f"{arg.dtype} {arg.id}" for arg in args))


def c_pointer_dtype(dtype):
    return dtype + "*"


def c_value_dtype(dtype):
    if dtype[-1] != "*":
        Meta_Toolbox_Error("invalid dtype")
    return dtype[:-1]
