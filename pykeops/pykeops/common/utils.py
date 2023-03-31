import fcntl
import functools
import importlib.util
import os

import pykeops.config

c_type = dict(float16="half2", float32="float", float64="double")


def axis2cat(axis):
    """
    Axis is the dimension to sum (the pythonic way). Cat is the dimension that
    remains at the end (the Keops way).
    :param axis: 0 or 1
    :return: cat: 1 or 0
    """
    if axis in [0, 1]:
        return (axis + 1) % 2
    else:
        raise ValueError("Axis should be 0 or 1.")


def cat2axis(cat):
    """
    Axis is the dimension to sum (the pythonic way). Cat is the dimension that
    remains at the end (the Keops way).
    :param cat: 0 or 1
    :return: axis: 1 or 0
    """
    if cat in [0, 1]:
        return (cat + 1) % 2
    else:
        raise ValueError("Category should be Vi or Vj.")


def get_tools(lang):
    """
    get_tools is used to simulate template as in Cpp code. Depending on the langage
    it import the right classes.

    :param lang: a string with the langage ('torch'/'pytorch' or 'numpy')
    :return: a class tools
    """

    if lang == "numpy":
        from pykeops.numpy.utils import numpytools

        tools = numpytools()
    elif lang == "torch" or lang == "pytorch":
        from pykeops.torch.utils import torchtools

        tools = torchtools()

    return tools


def WarmUpGpu(lang):
    tools = get_tools(lang)
    # dummy first calls for accurate timing in case of GPU use
    my_routine = tools.Genred(
        "SqDist(x,y)",
        ["x = Vi(1)", "y = Vj(1)"],
        reduction_op="Sum",
        axis=1,
        dtype=tools.dtype,
    )
    dum = tools.rand(10, 1)
    my_routine(dum, dum)
    my_routine(dum, dum)


def max_tuple(a, b):
    return tuple(max(a_i, b_i) for (a_i, b_i) in zip(a, b))


def check_broadcasting(dims_1, dims_2):
    r"""
    Checks that the shapes **dims_1** and **dims_2** are compatible with each other.
    """
    if dims_1 is None:
        return dims_2
    if dims_2 is None:
        return dims_1

    padded_dims_1 = (1,) * (len(dims_2) - len(dims_1)) + dims_1
    padded_dims_2 = (1,) * (len(dims_1) - len(dims_2)) + dims_2

    for (dim_1, dim_2) in zip(padded_dims_1, padded_dims_2):
        if dim_1 != 1 and dim_2 != 1 and dim_1 != dim_2:
            raise ValueError(
                "Incompatible batch dimensions: {} and {}.".format(dims_1, dims_2)
            )

    return max_tuple(padded_dims_1, padded_dims_2)


def pyKeOps_Message(message, use_tag=True, **kwargs):
    if pykeops.verbose:
        tag = "[pyKeOps] " if use_tag else ""
        message = tag + message
        print(message, **kwargs)


def pyKeOps_Warning(message):
    if pykeops.verbose:
        message = "[pyKeOps] Warning : " + message
        print(message)
