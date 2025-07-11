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
        cuda_bin = Path(cuda_path, "bin")  # where the dlls are located

        if find_library("nvcuda") is not None:
            output["dll_cuda"] = find_library("nvcuda")

        for file in cuda_bin.iterdir():

            if file.name.startswith("cudart") and file.name.endswith(".dll"):
                output["dll_cudart"] = str(file)

            if file.name.startswith("nvrtc-builtins") and file.name.endswith(".dll"):
                output["dll_nvrtc"] = str(file)

        # See the files in Path(cuda_path, "cmake") for something more automatic here
        cuda_include = Path(cuda_path, "include")
        if cuda_include.is_dir():
            output["include_dir"] = str(cuda_include)

        cuda_libs = Path(cuda_path, "lib", "x64")
        if cuda_libs.is_dir():
            output["lib_dirs"] = str(cuda_libs)

        output["lib_names"] = {}
        for key in ["cuda", "nvrtc", "cudart"]:

            if (cuda_libs / (key + ".lib")).is_file():
                output["lib_names"][key] = key

    return output
