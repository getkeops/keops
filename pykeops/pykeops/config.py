import importlib.util
import os
import sys

# Instantiating the keopscore.config main classes for pykeops
import keopscore.config

cuda = keopscore.config.cuda
path = keopscore.config.path
openmp = keopscore.config.openmp
cxx = keopscore.config.cxx
debug = keopscore.config.debug

get_build_folder = path.get_build_folder
# keep the old gpu_available variable for backward compatibility, but it is now recommended to use cuda.is_available() instead
gpu_available = cuda.is_available()
default_gpu_id = cuda.get_default_gpu_id()


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

base_dir_path = os.path.abspath(os.path.dirname(__file__))


def read_version(version_file):
    with open(version_file, encoding="utf-8") as v:
        return v.read().rstrip()


_version = read_version(os.path.join(base_dir_path, "keops_version"))


def get_version():
    assert (
        _version == keopscore.__version__
    ), f"Version mismatch between pykeops and keopscore: {_version} vs {keopscore.__version__}"
    return _version


# path
def get_pykeops_io_folder():
    return os.path.join(base_dir_path, "common", "keops_io")


def pykeops_nvrtc_name(type="src"):
    basename = "pykeops_nvrtc"
    if type == "src":
        return os.path.join(get_pykeops_io_folder(), "nvrtc", basename + ".cpp")
    return path.get_python_extension_path(basename)


def pykeops_cpp_binder_name(type="src"):
    basename = "pykeops_cpp"
    if type == "src":
        return os.path.join(get_pykeops_io_folder(), "cpp", basename + ".cpp")
    return path.get_python_extension_path(basename)


python_includes = "$({python3} -m pybind11 --includes)".format(python3=sys.executable)
