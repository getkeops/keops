import importlib.util
import os
from hashlib import sha256

from pykeops import bin_folder, build_type
from pykeops.common.compile_routines import compile_generic_routine
from pykeops.common.utils import module_exists, create_and_lock_build_folder


def TestChunkedTiles(formula):
    import re
    if re.search(".*Reduction\(Sum\((Square|Abs)\(\(Var\(.*?\) . Var\(.*?\)\)\)\).*?\)", formula) is not None:
        dim = [0, 0]
        for k in range(2):
            varstr = re.findall("Var\(.,.*?," + str(k) + "\)", formula)[0]
            loc = re.search(",.*?,", varstr).span()
            loc = loc[0] + 1, loc[1] - 1
            dim[k] = int(varstr[loc[0]:loc[1]])
        if dim[0] == dim[1] and dim[0] > 100:
            return True
    return False


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

        if TestChunkedTiles(formula):
            self.optional_flags += ['-DENABLECHUNK=1']

        # create the name from formula, aliases and dtype.
        self.dll_name = self._create_name(formula, aliases, dtype, lang, self.optional_flags)

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
