import importlib.util

import pykeops.config
from pykeops.common.compile_routines import compile_generic_routine
from pykeops.common.utils import module_exists, create_and_lock_build_folder
from pykeops.common.set_path import create_name, set_build_folder

   
    


class LoadKeOps:
    """
    Load the keops shared library that corresponds to the given formula, aliases, dtype and lang.
    If the shared library cannot be loaded, it will be compiled.
    Note: This function is thread/process safe by using a file lock.

    :return: The Python function that corresponds to the loaded Keops kernel.
    """

    def __init__(self, formula, aliases, dtype, lang, optional_flags=[]):
        self.formula = formula
        self.aliases = aliases
        self.dtype = dtype
        self.lang = lang
        self.optional_flags = optional_flags
        
        # create the name from formula, aliases and dtype.
        self.dll_name = create_name(self.formula, self.aliases, self.dtype, self.lang, self.optional_flags)

        if (not module_exists(self.dll_name)) or (pykeops.config.build_type == 'Debug'):
            self.build_folder = set_build_folder(pykeops.config.bin_folder, self.dll_name)
            self._safe_compile()

    @create_and_lock_build_folder()
    def _safe_compile(self):
        compile_generic_routine(self.formula, self.aliases, self.dll_name, self.dtype, self.lang,
                                self.optional_flags, self.build_folder)

    def import_module(self):
        return importlib.import_module(self.dll_name)
