import importlib.util
import sysconfig
from os.path import join
import keops.config

###############################################################
# Initialize some variables: the values may be redefined later

numpy_found = importlib.util.find_spec("numpy") is not None
torch_found = importlib.util.find_spec("torch") is not None

from keops.config.config import use_cuda as gpu_available

jit_binary_name = join(keops.config.config.build_path,
                       "keops_io_nvrtc" + sysconfig.get_config_var('EXT_SUFFIX')
                       )

python_includes = "$(python3 -m pybind11 --includes)"