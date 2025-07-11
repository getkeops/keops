from pathlib import Path

from .compile import compile
from .detection import (
    include_dirs,
    lib_dirs,
    lib_names,
)
from .utils import find_package_location


def compile_nvrtc_jit(build_folder):

    keops_dir = Path(find_package_location("keopscore")).parent
    source_file = keops_dir / "binders" / "nvrtc" / "nvrtc_jit_win.cpp"

    macros = [
        "-DMAXIDGPU=0",
        "-DMAXTHREADSPERBLOCK0=1024",
        "-DSHAREDMEMPERBLOCK0=49152",
        "-DnvrtcGetTARGET=nvrtcGetCUBIN",
        "-DnvrtcGetTARGETSize=nvrtcGetCUBINSize",
        '-DARCHTAG="sm"',
    ]

    compile(
        source_file=source_file,
        project_name="nvrtc_jit",
        macros=macros,
        includes=[include_dirs[key] for key in ["keops", "cuda"]],
        link_dirs=[lib_dirs[key] for key in ["cuda"]],
        links=[lib_names[key] for key in ["cuda", "nvrtc", "cudart"]],
        suffix=".dll",
        output_dir=build_folder,
        print_cmakelists=False,
        show_cmake_commands_output=False,
        clean_tmp_build_dir=False,
    )
