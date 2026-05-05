import importlib.util
import os
import sys
import sysconfig


# Instantiating the keopscore.config main classes for pykeops
import keopscore.config

cuda = keopscore.config.cuda
path = keopscore.config.path
openmp = keopscore.config.openmp
cxx = keopscore.config.cxx
debug = keopscore.config.debug

get_build_folder = path.get_build_folder
gpu_available = cuda.get_use_cuda()


# Initialize some variables: the values may be redefined later

numpy_found = importlib.util.find_spec("numpy") is not None
torch_found = importlib.util.find_spec("torch") is not None


# Verbosity level


class _VerboseConfig:
    def __init__(self, level=1):
        self.level = level


_verbose_state = _VerboseConfig(1)

def set_verbose(val):
    val = int(val)
    if val not in (0, 1, 2):
        raise ValueError("Verbose level must be 0, 1 or 2.")

    _verbose_state.level = val
    os.environ["KEOPS_VERBOSE"] = str(val)
    debug.set_verbose(val)
    return _verbose_state.level

def init_verbose():
    env_val = os.getenv("PYKEOPS_VERBOSE")
    if env_val in ("0", "1", "2"):
        return set_verbose(int(env_val))

    return set_verbose(1)


def get_verbose():
    return _verbose_state.level


# version
def read_version(version_file):
    with open(version_file, encoding="utf-8") as v:
        return v.read().rstrip()

_version = read_version(os.path.join(os.path.abspath(os.path.dirname(__file__)), "keops_version"))

def get_version():
    assert _version == keopscore.__version__, f"Version mismatch between pykeops and keopscore: {_version} vs {keopscore.__version__}"
    return _version

















def pykeops_nvrtc_name(type="src"):
    basename = "pykeops_nvrtc"
    extension = ".cpp" if type == "src" else sysconfig.get_config_var("EXT_SUFFIX")
    return os.path.join(
        (
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "common", "keops_io"
            )
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
