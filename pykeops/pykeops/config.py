import importlib.util
import sysconfig
from os.path import join, dirname, realpath

###############################################################
# Initialize some variables: the values may be redefined later

numpy_found = importlib.util.find_spec("numpy") is not None
torch_found = importlib.util.find_spec("torch") is not None

from keopscore.config.config import use_cuda as gpu_available
from keopscore.config.config import get_build_folder


def pykeops_nvrtc_name(type="src"):
    basename = "pykeops_nvrtc"
    extension = ".cpp" if type == "src" else sysconfig.get_config_var("EXT_SUFFIX")
    return join(
        join(dirname(realpath(__file__)), "common", "keops_io")
        if type == "src"
        else get_build_folder(),
        basename + extension,
    )


def pykeops_cpp_name(tag="", extension=""):
    basename = "pykeops_cpp_"
    return join(
        get_build_folder(),
        basename + tag + extension,
    )


python_includes = "$(python3 -m pybind11 --includes)"
