import sys
import sysconfig
from pathlib import Path

import pybind11

from .cuda_detection import detect_cuda_toolkit
from .utils import find_package_location

include_dirs = {}
lib_dirs = {}
lib_names = {}
dlls = {}


try:
    location_keops_init = find_package_location("keopscore")
    include_dirs["keops"] = str(Path(location_keops_init).parent)
    keops_available = True
except ImportError:
    keops_available = False



include_dirs["pybind11"] = pybind11.get_include()


# Python
include_dirs["python"] = sysconfig.get_path('include')

if sys.platform == "win32":
    # On Windows, get the path to the Python DLL
    python_libs = Path(sysconfig.get_path('include')).parent / "libs"

    # Get the path to the standard library (Lib)
    if python_libs.is_dir():
        lib_dirs["python"] = str(python_libs)

        version = str(sys.version_info.major) + str(sys.version_info.minor)
        if (Path(lib_dirs["python"]) / ("python" + version + ".lib")).is_file():
            lib_names["python"] = "python" + version
else:
    # On Unix-like systems, use sysconfig to get the library directory and name
    lib_dirs["python"] = Path(sysconfig.get_config_var('LIBDIR'))
    lib_names["python"] = sysconfig.get_config_var('LDLIBRARY')



# Cuda
cuda_config = detect_cuda_toolkit()

for key in ["cuda", "nvrtc", "cudart"]:

    if f"dll_{key}" in cuda_config:
        dlls[key] = cuda_config[f"dll_{key}"]

    if key in cuda_config["lib_names"]:
        lib_names[key] = cuda_config["lib_names"][key]

if "include_dir" in cuda_config:
    include_dirs["cuda"] = cuda_config["include_dir"]

if "lib_dirs" in cuda_config:
    lib_dirs["cuda"] = cuda_config["lib_dirs"]
