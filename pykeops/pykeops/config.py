import importlib.util
import os
import sys
import sysconfig

###############################################################
# Initialize some variables: the values may be redefined later

numpy_found = importlib.util.find_spec("numpy") is not None
torch_found = importlib.util.find_spec("torch") is not None

# Instantiating the keopscore.config main classes for pykeops
import keopscore.config

cuda = keopscore.config.cuda
path = keopscore.config.path
openmp = keopscore.config.openmp
cxx = keopscore.config.cxx

get_build_folder = path.get_build_folder
gpu_available = cuda.get_use_cuda()


def pykeops_nvrtc_name(type="src"):
    basename = "pykeops_nvrtc"
    extension = ".cpp" if type == "src" else sysconfig.get_config_var("EXT_SUFFIX")
    return os.path.join(
        (
            os.path.join(os.path.dirname(os.path.realpath(__file__)), "common", "keops_io")
            if type == "src"
            else get_build_folder()
        ),
        basename + extension,
    )


def pykeops_cpp_name(tag="", extension=""):
    basename = "pykeops_cpp_"
    return os.path.join(
        get_build_folder(),
        basename + tag + extension,
    )


python_includes = "$({python3} -m pybind11 --includes)".format(python3=sys.executable)
