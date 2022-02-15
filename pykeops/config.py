import importlib.util
import sysconfig
from os.path import join
import keops.config

###############################################################
# Initialize some variables: the values may be redefined later

numpy_found = importlib.util.find_spec("numpy") is not None
torch_found = importlib.util.find_spec("torch") is not None

from keops.config.config import use_cuda as gpu_available

def pykeops_nvrtc_name():
    basename = "pykeops_nvrtc"
    extension = sysconfig.get_config_var("EXT_SUFFIX")
    return join(
        keops.config.config.build_path,
        basename + extension,
        )

def pykeops_cpp_basename(tag=None):
    basename = "pykeops_cpp_"
    return join(
        keops.config.config.build_path,
        basename + tag,
        )

python_includes = "$(python3 -m pybind11 --includes)"
