import importlib.util
import os

import pykeops.config
from pykeops.common.compile_routines import (
    compile_generic_routine,
    get_pybind11_template_name,
    get_build_folder_name,
    check_or_prebuild,
)
from pykeops.common.utils import module_exists, create_and_lock_build_folder
from pykeops.common.set_path import create_name


class LoadKeOps:
    """
    Load the keops shared library that corresponds to the given formula, aliases, dtype and lang.
    If the shared library cannot be loaded, it will be compiled.
    Note: This function is thread/process safe by using a file lock.

    :return: The Python function that corresponds to the loaded Keops kernel.
    """

    def __init__(
        self, formula, aliases, dtype, lang, optional_flags=[], include_dirs=[]
    ):
        self.formula = formula
        self.aliases = aliases
        self.dtype = dtype
        self.lang = lang
        self.optional_flags = optional_flags
        self.include_dirs = include_dirs

        # get build folder name for dtype
        self.build_folder = get_build_folder_name(dtype, lang, include_dirs)

        # get template name for dtype
        self.template_name = get_pybind11_template_name(dtype, lang, include_dirs)

        # create the name from formula, aliases and dtype.
        self.dll_name = create_name(
            self.formula, self.aliases, self.dtype, self.lang, self.optional_flags
        )

        if (not module_exists(self.dll_name, self.template_name)) or (
            pykeops.config.build_type == "Debug"
        ):
            self._safe_compile()

    @create_and_lock_build_folder()
    def _safe_compile(self):
        # if needed, safely run cmake in build folder to prepare for building
        check_or_prebuild(self.dtype, self.lang, self.include_dirs)
        # launch compilation and linking of required KeOps formula
        compile_generic_routine(
            self.formula,
            self.aliases,
            self.dll_name,
            self.dtype,
            self.lang,
            self.optional_flags,
            self.include_dirs,
            self.build_folder,
        )

    def import_module(self):
        full_dll_name = self.dll_name + "." + self.template_name
        module_bin_folder = os.path.dirname(
            os.path.dirname(importlib.util.find_spec(full_dll_name).origin)
        )
        if not os.path.samefile(
            module_bin_folder,
            pykeops.config.bin_folder,
        ):
            raise ImportError(
                "[pyKeOps]: Current pykeops.config.bin_folder is {} but keops module {} is loaded from {} folder. "
                "This may happened when changing bin_folder **after** loading a keops module. Please check everything "
                "is fine.".format(
                    pykeops.config.bin_folder,
                    self.dll_name,
                    module_bin_folder,
                )
            )
        return importlib.import_module(full_dll_name)
