from pathlib import Path

from .compile import compile
from .detection import (
    include_dirs,
    lib_dirs,
    lib_names,
)
from .utils import find_package_location


def compile_pykeops_nvrtc(build_folder):

    pykeops_dir = Path(find_package_location("pykeops")).parent
    source_file = pykeops_dir / "common" / "keops_io" / "pykeops_nvrtc_win.cpp"

    macros = [
        "-DMAXIDGPU=0",  # TODO should be detected
        "-DMAXTHREADSPERBLOCK0=1024",  # TODO should be detected
        "-DSHAREDMEMPERBLOCK0=49152",  # TODO should be detected
        "-DnvrtcGetTARGET=nvrtcGetCUBIN",  # TODO should be detected
        "-DnvrtcGetTARGETSize=nvrtcGetCUBINSize",  # TODO should be detected
        '-DARCHTAG="sm"',  # TODO should be detected
    ]

    compile(
        source_file=source_file,
        project_name="pykeops_nvrtc",
        macros=macros,
        includes=[include_dirs[key] for key in ["python", "pybind11", "keops", "cuda"]],
        link_dirs=[lib_dirs[key] for key in ["python", "cuda"]],
        links=[lib_names[key] for key in ["cuda", "nvrtc", "cudart", "python"]],
        suffix=".pyd",
        output_dir=build_folder,
        print_cmakelists=False,
        show_cmake_commands_output=False,
        clean_tmp_build_dir=False,
    )
