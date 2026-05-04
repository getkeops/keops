import os
import subprocess
from ctypes import c_int, c_void_p, c_char_p, CDLL, byref, cast, POINTER, Structure
from ctypes.util import find_library

from keopscore.utils.messages import KeOps_Print, KeOps_Warning


def KeOps_OS_Run(command, print_warning=True):
    out = subprocess.run(command, shell=True, capture_output=True)
    if out.stderr != b"" and print_warning:
        KeOps_Warning(
            "There were warnings or errors while executing: " + command, newline=True
        )
        KeOps_Print(out.stderr.decode("utf-8"))
    return out


def get_include_file_abspath(filename, compiler):
    """Return the full path of the header filename using compiler."""

    cmd = f'echo "#include <{filename}>" | {compiler} -M -E -x c++ -'
    out = KeOps_OS_Run(cmd)

    text = out.stdout.decode("utf8")
    text = text.replace("\\\n", " ")

    try:
        deps = text.split(":", 1)[1]
    except IndexError:
        return None

    for path in deps.split():
        if os.path.basename(path) == filename:
            return path

    return None


def find_library_abspath(lib):
    """
    Wrapper around ctypes find_library that returns the full path of the library.
    Warning: it also opens the shared library.
    """

    class LINKMAP(Structure):
        _fields_ = [("l_addr", c_void_p), ("l_name", c_char_p)]

    res = find_library(lib)
    if res is None:
        return ""

    if os.path.isabs(res):
        return res

    lib = CDLL(res)
    libdl = CDLL(find_library("dl"))

    try:
        dlinfo = libdl.dlinfo
    except AttributeError:
        return ""
    dlinfo.argtypes = c_void_p, c_int, c_void_p
    dlinfo.restype = c_int

    lmptr = c_void_p()
    dlinfo(lib._handle, 2, byref(lmptr))

    abspath = cast(lmptr, POINTER(LINKMAP)).contents.l_name
    return abspath.decode("utf-8")


def _find_library_by_names(library_names):
    """Return the first library path resolved by ctypes for known library names."""
    for library_name in library_names:
        library_path = find_library(library_name)
        if library_path:
            return find_library_abspath(library_name)

    return None
