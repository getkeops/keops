import importlib.util
import os
from hashlib import sha256

from pykeops import bin_folder, build_type
from pykeops.common.compile_routines import compile_generic_routine
from pykeops.common.utils import module_exists, create_and_lock_build_folder


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
        self.dll_name = self._create_name(formula, aliases, dtype, lang, optional_flags)

        if (module_exists(self.dll_name)) or (build_type == 'Debug'):
            self.build_folder = os.path.join(bin_folder, 'build-' + self.dll_name)
            self._safe_compile()

    def _create_name(self, formula, aliases, dtype, lang, optional_flags):
        """
        Compose the shared object name
        """
        formula = formula.replace(" ", "")  # Remove spaces
        aliases = [alias.replace(" ", "") for alias in aliases]
    
        # Since the OS prevents us from using arbitrary long file names, an okayish solution is to call
        # a standard hash function, and hope that we won't fall into a non-injective nightmare case...
        dll_name = ",".join(aliases + [formula] + optional_flags) + "_" + dtype
        dll_name = "libKeOps" + lang + sha256(dll_name.encode("utf-8")).hexdigest()[:10]
        return dll_name

    @create_and_lock_build_folder()
    def _safe_compile(self):
        compile_generic_routine(self.formula, self.aliases, self.dll_name, self.dtype, self.lang,
                                self.optional_flags, build_folder=self.build_folder)

    def import_module(self):
        return importlib.import_module(self.dll_name)
