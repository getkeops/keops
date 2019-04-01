import fcntl
import functools
from hashlib import sha256

c_type = dict(float32="float", float64="double")

def create_name(formula, aliases, dtype, lang):
    """
    Compose the shared object name
    """
    formula = formula.replace(" ", "")  # Remove spaces
    aliases = [alias.replace(" ", "") for alias in aliases]

    # Since the OS prevents us from using arbitrary long file names, an okayish solution is to call
    # a standard hash function, and hope that we won't fall into a non-injective nightmare case...
    dll_name = ",".join(aliases + [formula]) + "_" + dtype
    dll_name = "libKeOps" + lang + sha256(dll_name.encode("utf-8")).hexdigest()[:10]
    return dll_name


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


class FileLock:
    def __init__(self, fd, op=fcntl.LOCK_EX):
        self.fd = fd
        self.op = op

    def __enter__(self):
        fcntl.flock(self.fd, self.op)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        fcntl.flock(self.fd, fcntl.LOCK_UN)


def filelock(build_folder, lock_file_name="pykeops.lock"):
    """
    This function is used to lock the building dir (see cmake) too avoid two concurrency
    threads using the same cache files.
    """
    def wrapper(func):
        @functools.wraps(func)
        def wrapper_filelock(*args, **kwargs):
            with open(build_folder + '/' + lock_file_name, 'w') as f:
                with FileLock(f):
                    return func(*args, **kwargs)

        return wrapper_filelock
    
    return wrapper


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


def WarmUpGpu(backend):
    tools = get_tools(backend)
    
    # dummy first calls for accurate timing in case of GPU use
    formula = "Exp(-oos2*SqDist(x,y))*b"
    variables = ["x = Vi(1)",  # First arg : i-variable, of size 1
                 "y = Vj(1)",  # Second arg: j-variable, of size 1
                 "b = Vj(1)",  # Third arg : j-variable, of size 1
                 "oos2 = Pm(1)"]  # Fourth arg: scalar parameter
    
    my_routine = tools.Genred(formula, variables, reduction_op='Sum', axis=1, dtype=tools.dtype)
    dum = tools.rand(10, 1)
    dum2 = tools.rand(10, 1)
    my_routine(dum, dum, dum2, tools.array([1.0]))
    my_routine(dum, dum, dum2, tools.array([1.0]))
