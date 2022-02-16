import sysconfig
import importlib.util
from os.path import join, dirname, realpath
import keops.config

###############################################################
# Initialize some variables: the values may be redefined later

numpy_found = importlib.util.find_spec("numpy") is not None
torch_found = importlib.util.find_spec("torch") is not None

from keops.config.config import use_cuda as gpu_available


def pykeops_nvrtc_name(type="src"):
    basename = "pykeops_nvrtc"
    extension = ".cpp" if type == "src" else sysconfig.get_config_var("EXT_SUFFIX")
    return join(
        join(dirname(realpath(__file__)), "common", "keops_io")
        if type == "src"
        else keops.config.config.build_path,
        basename + extension,
    )


def pykeops_cpp_name(tag="", extension=""):
    basename = "pykeops_cpp_"
    return join(keops.config.config.build_path, basename + tag + extension,)


python_includes = "$(python3 -m pybind11 --includes)"
