import importlib.util
import os

import pykeops.config
from pykeops.common.compile_routines import compile_generic_routine
from pykeops.common.utils import module_exists, create_and_lock_build_folder
from pykeops.common.set_path import create_name, set_build_folder


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
        self.dll_name = create_name(self.formula, self.aliases, self.dtype, self.lang, self.optional_flags)

        if (not module_exists(self.dll_name)) or (pykeops.config.build_type == 'Debug'):
            self.build_folder = set_build_folder(pykeops.config.bin_folder, self.dll_name)
            self._safe_compile()

    @create_and_lock_build_folder()
    def _safe_compile(self):
        compile_generic_routine(self.formula, self.aliases, self.dll_name, self.dtype, self.lang,
                                self.optional_flags, self.build_folder)

    def import_module(self):
        # if not os.path.samefile(os.path.dirname(importlib.util.find_spec(self.dll_name).origin),
        #                         pykeops.config.bin_folder):
        #     raise ImportError(
        #         "[pyKeOps]: Current pykeops.config.bin_folder is {} but keops module {} is loaded from {} folder. "
        #         "This may happened when changing bin_folder **after** loading a keops module. Please check everything "
        #         "is fine.".format(
        #             pykeops.config.bin_folder, self.dll_name,
        #             os.path.dirname(importlib.util.find_spec(self.dll_name).origin)))
        return importlib.import_module(self.dll_name)
