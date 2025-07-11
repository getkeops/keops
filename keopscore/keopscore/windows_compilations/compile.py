from __future__ import annotations

import os
import shutil
import sysconfig
import uuid
from pathlib import Path

from .globals import tmp_dir

_empty_list = []


def compile(
    source_file: os.PathLike,
    project_name: str | None = None,
    includes: list[os.PathLike] | None = _empty_list,
    link_dirs: list[os.PathLike] | None = _empty_list,
    links: list[str] | None = _empty_list,
    macros: list[str] | None = _empty_list,
    suffix: str = ".dll",
    output_dir=".",
    print_cmakelists=False,
    show_cmake_commands_output=False,
    clean_tmp_build_dir=True,
):

    output_dir = Path(output_dir)

    if project_name is None:
        project_name = "".join(Path(source_file).name.split(".")[:-1])

    includes_str = ""
    for include in includes:
        includes_str += f'include_directories("{include!s}")\n'
    includes_str = includes_str.replace("\\", "/")

    link_dirs_str = ""
    for link in link_dirs:
        link_dirs_str += f'link_directories("{link!s}")\n'
    link_dirs_str = link_dirs_str.replace("\\", "/")

    macros_str = ""
    for macro in macros:
        macros_str += f"add_definitions({macro!s})\n"

    if len(links) == 0:
        links = ""

    else:
        inline_links = " ".join(links)
        links = f"target_link_libraries(${{PROJECT_NAME}} {inline_links})"

    with Path.open(Path(__file__).parent / "templates" / "CMakeLists.txt") as f:
        content = f.read()

    includes = includes_str
    link_dirs = link_dirs_str
    macros = macros_str
    source_file = str(Path(source_file).resolve()).replace("\\", "/")

    if "pyd" in suffix:
        suffix = sysconfig.get_config_var("EXT_SUFFIX")

    fields = [
        "source_file",
        "project_name",
        "includes",
        "link_dirs",
        "links",
        "macros",
        "suffix",
    ]

    for field in fields:
        content = content.replace(f"***{field}***", locals()[field])

    import os
    import subprocess

    cwd = Path.cwd()

    # Go to tmp dir

    tmp_build = tmp_dir / str(uuid.uuid4())

    tmp_build.mkdir()
    if (tmp_build / "build").is_dir():
        shutil.rmtree(tmp_build / "build")

    tmp_build.mkdir(exist_ok=True)

    with Path.open(Path(tmp_build) / "CMakeLists.txt", "w") as f:
        f.write(content)

    if print_cmakelists:
        print(content)

    os.chdir(tmp_build)
    Path("build").mkdir()
    os.chdir("build")

    if not show_cmake_commands_output:

        with Path.open(tmp_build / "log", "w") as log_file:

            subprocess.check_call(["cmake", ".."], stdout=log_file)
            subprocess.check_call(
                ["cmake", "--build", ".", "--config", "Release"], stdout=log_file
            )

    else:
        subprocess.check_call(["cmake", ".."])
        subprocess.check_call(["cmake", "--build", ".", "--config", "Release"])

    # Back to previous working directory
    os.chdir(cwd)

    # Define the source and destination directories
    Path(output_dir).mkdir(exist_ok=True)

    source_dir = tmp_build / "build" / "Release"

    # Copy the contents of the source directory to the destination directory
    for item in os.listdir(source_dir):
        s = source_dir / item
        d = output_dir / item

        # Copy files or directories
        if str(s).endswith(suffix):
            shutil.copy2(s, d)

    if clean_tmp_build_dir:
        shutil.rmtree(tmp_build)
