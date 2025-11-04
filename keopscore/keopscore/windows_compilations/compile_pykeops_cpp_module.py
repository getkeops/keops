from .compile import compile
from .detection import include_dirs, lib_dirs, lib_names


def compile_pykeops_cpp_module(source_file, build_folder):
    """This built-in compilation config serves to compile pykeops cpp modules

    Parameters
    ----------
    source_file
        Location of the source cpp file
    build_folder
        The KeOps build folder

    """

    compile(
        source_file=source_file,
        includes=[
            include_dirs["python"],
            include_dirs["pybind11"],
            include_dirs["keops"],
        ],
        link_dirs=[lib_dirs["python"]],
        links=[lib_names["python"]],
        suffix=".pyd",
        output_dir=build_folder,
        print_cmakelists=False,
        show_cmake_commands_output=False,
        clean_tmp_build_dir=False,
    )
