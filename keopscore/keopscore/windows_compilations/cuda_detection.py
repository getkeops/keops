r"""
CUDA toolkit detection on Windows.

CUDA_PATH environment variable must be set. It is usually set by the CUDA installer, if not it
must point to a valid CUDA installation (typically C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.Y).

The detection looks for the following files:
- cudart*.dll
- nvrtc-builtins*.dll
- nvcuda.dll (CUDA driver library, usually located in system32 folder)
- include directory
- lib/x64 directory containing cuda.lib, nvrtc.lib and cudart.lib
"""

import os
from ctypes.util import find_library
from pathlib import Path

cuda_available = "CUDA_PATH" in os.environ


def detect_cuda_toolkit():

    output = {}

    if cuda_available:

        cuda_path = Path(
            os.environ["CUDA_PATH"]
        )  # base path for cuda installation (including bin, lib, include, etc.)

        if find_library("nvcuda") is not None:  # NVCUDA is the main CUDA driver library
            output["dll_cuda"] = find_library("nvcuda")

        cuda_path = Path(
            os.environ["CUDA_PATH"]
        )  # base path for cuda installation (including bin, lib, include, etc.)

        #################################################
        # Detect relevant DLLs: cudart and nvrtc-builtins
        #################################################

        # Check both bin and bin/x64 directories for relevant DLLs
        bin_dirs = [Path(cuda_path, "bin"), Path(cuda_path, "bin", "x64")]

        for bin_dir in bin_dirs:
            if bin_dir.is_dir():
                for file in bin_dir.iterdir():
                    if file.name.startswith("cudart") and file.name.endswith(".dll"):
                        output["dll_cudart"] = str(file)
                    if file.name.startswith("nvrtc-builtins") and file.name.endswith(
                        ".dll"
                    ):
                        output["dll_nvrtc"] = str(file)

        #################################################
        # Detect include and lib directories
        #################################################
        cuda_include = Path(cuda_path, "include")
        if cuda_include.is_dir():
            output["include_dir"] = str(cuda_include)

        cuda_libs = Path(cuda_path, "lib", "x64")
        if cuda_libs.is_dir():
            output["lib_dirs"] = str(cuda_libs)

        #################################################
        # Make sure that cudart, nvrtc and cuda libs are available
        #################################################
        output["lib_names"] = {}
        for key in ["cuda", "nvrtc", "cudart"]:

            if (cuda_libs / (key + ".lib")).is_file():
                output["lib_names"][key] = key

    return output
