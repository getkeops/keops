import re
from collections import OrderedDict

from pykeops.common.utils import pyKeOps_Message

categories = OrderedDict([("Vi", 0), ("Vj", 1), ("Pm", 2)])


def complete_aliases(formula, aliases):
    """
    This function parse formula (a string) to find pattern like 'Var(x,x,x)'.
    It then append to aliases (list of strings), the extra 'Var(x,x,x)'.
    """
    # first we detect all instances of Var(*,*,*) in formula.
    # These may be extra variables that are not listed in the aliases
    extravars = re.findall(r"Var\([0-9]+,[0-9]+,[0-9]+\)", formula.replace(" ", ""))
    # we get unicity
    extravars = list(set(extravars))
    # now we loop through extravars
    newind = ()  # this will give the indices in extravars list of new variables
    newpos = ()  # this will give the indices in aliases list of new variables
    for (ind, var) in enumerate(extravars):
        # we get the "position" of the variable as the first integer value in the string
        # (i.e. the "a" in "Var(a,b,c)")
        pos = int(re.search(r"[0-9]+", var).group(0))
        if pos < len(aliases):
            # this means that in fact var is not an extra variable, it is already in the list of aliases
            # We could check that the "dimension" and "category" are consistent, but we assume here
            # that the formula is consistent. The check will be done in the C++ code.
            pass
        else:
            # we need to append var to aliases, but with correct position, so we first record the indices
            newind += (ind,)
            newpos += (pos,)
    # finally we append the new variables with correct ordering to the aliases list. We assume here again
    # that formula is consistent, more precisely
    # that pos is a permutation of len(aliases):len(aliases)+len(newind)
    aliases += [None] * len(newind)
    for i in range(len(newind)):
        aliases[newpos[i]] = extravars[newind[i]]
    return aliases


def get_sizes(aliases, *args):
    nx, ny = None, None
    for (var_ind, sig) in enumerate(aliases):
        _, cat, dim, pos = get_type(sig, position_in_list=var_ind)
        if cat == 0:
            nx = args[pos].shape[-2]
        elif cat == 1:
            ny = args[pos].shape[-2]
        if (nx is not None) and (ny is not None):
            return nx, ny

    # At this point, we know that our formula is degenerate,
    # with no "x" or no "y" variable. The sensible behavior is to assume that
    # the corresponding "empty" dimension is equal to 1,
    # in accordance with the dimcheck of keops_io.h: TODO: check the new Size !
    if nx is None:
        nx = 1
    if ny is None:
        ny = 1

    return nx, ny


def get_type(type_str, position_in_list=None):
    """
    Get the type of the variable declared in type_str.

    :param type_str: is a string of the form
    "var = Xy(dim)" or "var = Xy(pos,dim)" or "Xy(pos,dim)" or "Xy(dim)" with Xy being either Vi, Vj, Vx, Vy, or Pm,
    or "var = Var(pos,dim,cat)" (N.B. Vx and Vy are equivalent to resp. Vi and Vj and kept for backward compatibility)
    :param position_in_list: an optional integer used if the position is not given
                             in type_str (ie is of the form "var = Xy(dim)" or "Xy(dim)")

    :return: name : a string (here "var"), cat : an int (0,1 or 2), dim : an int
    """

    # switch old Vx Vy syntax to Vi Vj
    if ("Vx" in type_str) or ("Vy" in type_str):
        type_str = type_str.replace("Vx", "Vi")
        type_str = type_str.replace("Vy", "Vj")
        import warnings

        warnings.warn("'Vx' and 'Vy' variables types are now renamed 'Vi' and 'Vj'")

    m = re.match(
        r"([a-zA-Z_][a-zA-Z_0-9]*)=(Vi|Vj|Pm)\(([0-9]*?),?([0-9]*)\)",
        type_str.replace(" ", ""),
    )

    if m is None:
        m = re.match(r"(Vi|Vj|Pm)\(([0-9]*?),?([0-9]*)\)", type_str.replace(" ", ""))
        if m is None:
            m = re.match(
                r"Var\(([0-9]*?),?([0-9]*),?([0-9]*)\)", type_str.replace(" ", "")
            )
            if m is None:
                raise ValueError(
                    type_str
                    + " type_str does not match the 'var = [Vi|Vj|Pm](dim)' or 'var = [Vi|Vj|Pm](pos,dim) or '[Vi|Vj|Pm](dim) or '[Vi|Vj|Pm](pos,dim) or Var(pos,dim,cat)'  format: "
                    + type_str
                )
            else:
                # output: varname,          cat          ,     dim        , pos
                return None, int(m.group(3)), int(m.group(2)), int(m.group(1))
        else:
            # Try to infer position
            if m.group(2):
                pos = int(m.group(2))
            elif position_in_list is not None:
                pos = int(position_in_list)
            else:
                pos = None
            # output: varname,          cat          ,     dim        , pos
            return None, categories[m.group(1)], int(m.group(3)), pos
    else:
        # Try to infer position
        if m.group(3):
            pos = int(m.group(3))
        elif position_in_list is not None:
            pos = int(position_in_list)
        else:
            pos = None
        # output: varname,          cat          ,     dim        , pos
        return m.group(1), categories[m.group(2)], int(m.group(4)), pos


def get_optional_flags(
    reduction_op_internal, dtype_acc, use_double_acc, sum_scheme, enable_chunks
):
    # 1. Options for accuracy

    if dtype_acc != "auto" and use_double_acc:
        raise ValueError(
            pyKeOps_Message("you cannot set both options use_double_acc and dtype_acc.")
        )
    if use_double_acc:
        dtype_acc = "float64"
    if dtype_acc != "auto" and reduction_op_internal not in (
        "Sum",
        "Max_SumShiftExp",
        "Max_SumShiftExpWeight",
    ):
        raise ValueError(
            pyKeOps_Message(
                "parameter dtype_acc should be set to 'auto' for no-sum type reductions (Min, Max, ArgMin, etc.)"
            )
        )

    if sum_scheme == "auto":
        if reduction_op_internal in ("Sum", "Max_SumShiftExp", "Max_SumShiftExpWeight"):
            sum_scheme = "block_sum"
        else:
            sum_scheme = "direct_sum"
    if sum_scheme == "block_sum":
        if reduction_op_internal not in (
            "Sum",
            "Max_SumShiftExp",
            "Max_SumShiftExpWeight",
        ):
            raise ValueError(
                pyKeOps_Message(
                    'sum_scheme="block_sum" is only valid for sum type reductions.'
                )
            )
    elif sum_scheme == "kahan_scheme":
        if reduction_op_internal not in (
            "Sum",
            "Max_SumShiftExp",
            "Max_SumShiftExpWeight",
        ):
            raise ValueError(
                pyKeOps_Message(
                    'sum_scheme="kahan_scheme" is only valid for sum type reductions.'
                )
            )
    elif sum_scheme != "direct_sum":
        raise ValueError(
            pyKeOps_Message(
                'invalid value for option sum_scheme : should be one of "auto", "direct_sum", "block_sum" or "kahan_scheme".'
            )
        )

    optional_flags = dict()

    optional_flags["dtype_acc"] = dtype_acc

    optional_flags["sum_scheme"] = sum_scheme

    # 2. Option for chunk mode

    if enable_chunks:
        optional_flags["enable_chunks"] = 1
    else:
        optional_flags["enable_chunks"] = 0

    return optional_flags


def parse_dtype_acc(dtype_acc, dtype):

    if dtype_acc == "auto":
        dtype_acc = dtype
    if dtype == "float32" and dtype_acc not in ("float32", "float64"):
        raise ValueError(
            "[KeOps] invalid parameter dtype_acc : should be either 'float32' or 'float64' when dtype is 'float32'"
        )
    elif dtype == "float16" and dtype_acc not in ("float16", "float32"):
        raise ValueError(
            "[KeOps] invalid parameter dtype_acc : should be either 'float16' or 'float32' when dtype is 'float16'"
        )
    elif dtype == "float64" and dtype_acc not in "float64":
        raise ValueError(
            "[KeOps] invalid parameter dtype_acc : should be 'float64' when dtype is 'float64'"
        )

    if dtype_acc == "float64":
        dtype_acc = "double"
    elif dtype_acc == "float32":
        if dtype == "float16":
            dtype_acc = "float2"
        else:
            dtype_acc = "float"
    elif dtype_acc == "float16":
        dtype_acc = "half2"
    else:
        raise ValueError(
            '[KeOps] invalid value for option dtype_acc : should be one of "auto", "float16", "float32" or "float64".'
        )

    return dtype_acc
