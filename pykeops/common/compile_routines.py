import os
import pathlib
import shutil
import sys
from hashlib import sha256

import pykeops.config
from pykeops.common.parse_type import check_aliases_list
from pykeops.common.utils import c_type, replace_strings_in_file, run_and_display


def get_pybind11_template_name_and_command(dtype, lang, include_dirs):
    command_line = [
        "cmake",
        pykeops.config.script_template_folder,
        "-DCMAKE_BUILD_TYPE=" + "'{}'".format(pykeops.config.build_type),
        "-DPYTHON_LANG=" + "'{}'".format(lang),
        "-D__TYPE__=" + "'{}'".format(c_type[dtype]),
        "-DPYTHON_EXECUTABLE=" + "'{}'".format(sys.executable),
        "-DPYBIND11_PYTHON_VERSION="
        + "'{}'".format(
            str(sys.version_info.major) + "." + str(sys.version_info.minor)
        ),
        "-DC_CONTIGUOUS=1",
    ] + include_dirs
    template_name = (
        "libKeOps_template_"
        + sha256("".join(command_line).encode("utf-8")).hexdigest()[:10]
    )
    return template_name, command_line


def get_pybind11_template_name(dtype, lang, include_dirs):
    template_name, command_line = get_pybind11_template_name_and_command(
        dtype, lang, include_dirs
    )
    return template_name


def get_or_build_pybind11_template(
    dtype, lang, include_dirs, use_prebuilt_formula=False
):
    template_name, command_line = get_pybind11_template_name_and_command(
        dtype, lang, include_dirs
    )
    template_build_folder = (
        pykeops.config.bin_folder
        + os.path.sep
        + "build-pybind11_template-"
        + template_name
    )

    is_rebuilt = False
    if not os.path.exists(template_build_folder + os.path.sep + "CMakeCache.txt"):
        is_rebuilt = True
        print(
            "[pyKeOps] Compiling pybind11 template "
            + template_name
            + " in "
            + os.path.realpath(pykeops.config.bin_folder)
            + " ... ",
            end="",
            flush=True,
        )
        # print('(with dtype=',dtype,', lang=',lang,', include_dirs=',include_dirs,')', flush=True)

        os.mkdir(template_build_folder)

        command_line += ["-Dtemplate_name=" + "'{}'".format(template_name)]
        command_line += [
            "-Dkeops_formula_name=" + "'{}'".format(pykeops.config.shared_obj_name)
        ]

        build_folder = check_or_prebuild(dtype, lang, include_dirs)

        if not use_prebuilt_formula:
            # here we build an arbitrary dummy formula to create a formula object file ; otherwise the initial cmake call would fail
            formula = "Sum_Reduction(Var(0,1,0)*Var(1,1,1),0)"
            alias_string = ""
            optional_flags = []
            build_keops_formula_object_file(
                build_folder, dtype, formula, alias_string, optional_flags
            )

        fname = pykeops.config.shared_obj_name + ".o"
        os.rename(
            pykeops.config.bin_folder + os.path.sep + fname,
            template_build_folder + os.path.sep + fname,
        )

        run_and_display(
            command_line + ["-DcommandLine=" + " ".join(command_line)],
            template_build_folder,
            msg="CMAKE",
        )

        run_and_display(
            ["cmake", "--build", ".", "--target", template_name, "--", "VERBOSE=1"],
            template_build_folder,
            msg="MAKE",
        )

        print("done.", flush=True)

    return template_name, is_rebuilt


def create_keops_include_file(
    build_folder, dtype, formula, alias_string, optional_flags
):
    # creating KeOps include file for formula
    template_include_file = (
        pykeops.config.script_formula_folder + os.path.sep + "formula.h.in"
    )
    target_include_file = (
        build_folder + os.path.sep + pykeops.config.shared_obj_name + ".h"
    )
    shutil.copyfile(template_include_file, target_include_file)

    optional_flags_string = ""
    for flag in optional_flags:
        if flag[:2] != "-D":
            raise ValueError('Optional flag must be in the form "-D..."')
        macroid, macroval = str.split(flag[2:], "=")
        optional_flags_string += "#define " + macroid + " " + macroval + "\n"

    replace_pairs = [
        ("@__TYPE__@", str(c_type[dtype])),
        ("@FORMULA_OBJ@", formula),
        ("@VAR_ALIASES@", alias_string),
        ("@OPTIONAL_FLAGS@", optional_flags_string),
    ]
    replace_strings_in_file(target_include_file, replace_pairs)


def get_build_folder_name_and_command(dtype, lang, include_dirs):
    command_line = [
        "cmake",
        pykeops.config.script_formula_folder,
        "-DCMAKE_BUILD_TYPE=" + "'{}'".format(pykeops.config.build_type),
        "-Dshared_obj_name=" + "'{}'".format(pykeops.config.shared_obj_name),
        "-DPYTHON_LANG=" + "'{}'".format(lang),
        "-D__TYPE__=" + "'{}'".format(c_type[dtype]),
        "-DC_CONTIGUOUS=1",
    ] + include_dirs
    build_folder = (
        pykeops.config.bin_folder
        + os.path.sep
        + "build-"
        + sha256("".join(command_line).encode("utf-8")).hexdigest()[:10]
    )
    return build_folder, command_line


def get_build_folder_name(dtype, lang, include_dirs):
    build_folder, _ = get_build_folder_name_and_command(dtype, lang, include_dirs)
    return build_folder


def check_or_prebuild(dtype, lang, include_dirs):
    build_folder, command_line = get_build_folder_name_and_command(
        dtype, lang, include_dirs
    )
    if not os.path.exists(build_folder + os.path.sep + "CMakeCache.txt"):
        print(
            "[pyKeOps] Initializing build folder for dtype="
            + str(dtype)
            + " and lang="
            + lang
            + " in "
            + os.path.realpath(pykeops.config.bin_folder)
            + " ... ",
            end="",
            flush=True,
        )
        run_and_display(
            command_line + ["-DcommandLine=" + " ".join(command_line)],
            build_folder,
            msg="CMAKE",
        )
        print("done.", flush=True)
    return build_folder


def build_keops_formula_object_file(
    build_folder, dtype, formula, alias_string, optional_flags
):
    create_keops_include_file(
        build_folder, dtype, formula, alias_string, optional_flags
    )
    run_and_display(
        [
            "cmake",
            "--build",
            ".",
            "--target",
            pykeops.config.shared_obj_name,
            "--",
            "VERBOSE=1",
        ],
        build_folder,
        msg="MAKE",
    )


def compile_generic_routine(
    formula, aliases, dllname, dtype, lang, optional_flags, include_dirs, build_folder
):
    aliases = check_aliases_list(aliases)

    def process_alias(alias):
        if alias.find("=") == -1:
            return ""  # because in this case it is not really an alias, the variable is just named
        else:
            return "auto " + str(alias) + "; "

    def process_disp_alias(alias):
        return str(alias) + "; "

    alias_string = "".join([process_alias(alias) for alias in aliases])
    alias_disp_string = "".join([process_disp_alias(alias) for alias in aliases])

    print(
        "[pyKeOps] Compiling "
        + dllname
        + " in "
        + os.path.realpath(build_folder + os.path.sep + "..")
        + ":\n"
        + "       formula: "
        + formula
        + "\n"
        + "       aliases: "
        + alias_disp_string
        + "\n"
        + "       dtype  : "
        + dtype
        + "\n... ",
        flush=True,
    )

    build_keops_formula_object_file(
        build_folder, dtype, formula, alias_string, optional_flags
    )

    template_name, is_rebuilt = get_or_build_pybind11_template(
        dtype, lang, include_dirs, use_prebuilt_formula=True
    )

    template_build_folder = (
        pykeops.config.bin_folder
        + os.path.sep
        + "build-pybind11_template-"
        + template_name
    )

    if not is_rebuilt:
        fname = pykeops.config.shared_obj_name + ".o"
        os.rename(
            pykeops.config.bin_folder + os.path.sep + fname,
            template_build_folder + os.path.sep + fname,
        )
        run_and_display(
            ["cmake", "--build", ".", "--target", template_name, "--", "VERBOSE=1"],
            template_build_folder,
            msg="MAKE",
        )

    os.makedirs(pykeops.config.bin_folder + os.path.sep + dllname, exist_ok=True)
    fname = list(pathlib.Path(template_build_folder).glob(template_name + "*.so"))[
        0
    ].name
    os.rename(
        template_build_folder + os.path.sep + fname,
        pykeops.config.bin_folder + os.path.sep + dllname + os.path.sep + fname,
    )

    print("Done.", flush=True)


def compile_specific_conv_routine(dllname, dtype, build_folder):
    print(
        "Compiling "
        + dllname
        + " using "
        + dtype
        + " in "
        + os.path.realpath(build_folder + os.path.sep + "..")
        + "... ",
        end="",
        flush=True,
    )
    run_and_display(
        [
            "cmake",
            pykeops.config.script_specific_folder,
            "-DCMAKE_BUILD_TYPE=" + pykeops.config.build_type,
            "-Ushared_obj_name",
            "-D__TYPE__=" + c_type[dtype],
        ],
        build_folder,
        msg="CMAKE",
    )

    run_and_display(
        ["cmake", "--build", ".", "--target", dllname, "--", "VERBOSE=1"],
        build_folder,
        msg="MAKE",
    )
    print("Done.")


def compile_specific_fshape_scp_routine(
    dllname, kernel_geom, kernel_sig, kernel_sphere, dtype, build_folder
):
    print(
        "Compiling "
        + dllname
        + " using "
        + dtype
        + " in "
        + os.path.realpath(build_folder + os.path.sep + "..")
        + "... ",
        end="",
        flush=True,
    )
    run_and_display(
        [
            "cmake",
            pykeops.config.script_specific_folder,
            "-DCMAKE_BUILD_TYPE=" + pykeops.config.build_type,
            "-Ushared_obj_name",
            "-DKERNEL_GEOM=" + kernel_geom,
            "-DKERNEL_SIG=" + kernel_sig,
            "-DKERNEL_SPHERE=" + kernel_sphere,
            "-D__TYPE__=" + c_type[dtype],
        ],
        build_folder,
        msg="CMAKE",
    )

    run_and_display(
        ["cmake", "--build", ".", "--target", dllname, "--", "VERBOSE=1"],
        build_folder,
        msg="MAKE",
    )
    print("Done.")
