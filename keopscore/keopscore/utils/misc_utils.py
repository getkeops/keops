#######################################################################
# .  Warnings, Errors, etc.
#######################################################################

import keopscore

def KeOps_Message(message, use_tag=True, **kwargs):
    if keopscore.verbose:
        tag = "[KeOps] " if use_tag else ""
        message = tag + message
        print(message, **kwargs)


def KeOps_Warning(message, newline=False):
    if keopscore.verbose:
        message = ("\n" if newline else "") + "[KeOps] Warning : " + message
        print(message)


def KeOps_Error(message, show_line_number=True):
    message = "[KeOps] Error : " + message
    if show_line_number:
        from inspect import currentframe, getframeinfo

        frameinfo = getframeinfo(currentframe().f_back)
        message += f" (error at line {frameinfo.lineno} in file {frameinfo.filename})"
    raise ValueError(message)


def KeOps_OS_Run(command):
    import sys

    python_version = sys.version_info
    if python_version >= (3, 7):
        import subprocess, os

        out = subprocess.run(command, shell=True, capture_output=True)
        if out.stderr != b"":
            if os.name == 'nt':
                KeOps_Warning("There were warnings or errors compiling formula :", newline=True)
                print(out.stderr.decode("gbk"))
            else:
                KeOps_Warning("There were warnings or errors compiling formula :", newline=True)
                print(out.stderr.decode("utf-8"))
    elif python_version >= (3, 5):
        import subprocess

        subprocess.run(
            command,
            shell=True,
        )
    else:
        import os

        os.system(command)


def find_library_abspath_linux(lib):
    """
    wrapper around ctypes find_library that returns the full path
    of the library.
    Warning : it also opens the shared library !
    Adapted from
    https://stackoverflow.com/questions/35682600/get-absolute-path-of-shared-library-in-python/35683698
    """
    from ctypes import c_int, c_void_p, c_char_p, CDLL, byref, cast, POINTER, Structure
    from ctypes.util import find_library

    # linkmap structure, we only need the second entry
    class LINKMAP(Structure):
        _fields_ = [("l_addr", c_void_p), ("l_name", c_char_p)]

    res = find_library(lib)
    if res is None:
        return ""

    lib = CDLL(res)
    libdl = CDLL(find_library("dl"))

    dlinfo = libdl.dlinfo
    dlinfo.argtypes = c_void_p, c_int, c_void_p
    dlinfo.restype = c_int

    # gets typecasted later, I dont know how to create a ctypes struct pointer instance
    lmptr = c_void_p()

    # 2 equals RTLD_DI_LINKMAP, pass pointer by reference
    dlinfo(lib._handle, 2, byref(lmptr))

    # typecast to a linkmap pointer and retrieve the name.
    abspath = cast(lmptr, POINTER(LINKMAP)).contents.l_name

    return abspath.decode("utf-8")

def find_library_abspath_windows(lib):
    """
    wrapper around ctypes find_library that returns the full path
    of the library.
    Warning : it also opens the shared library !
    Adapted from
    https://stackoverflow.com/questions/11007896/how-can-i-search-and-get-the-directory-of-a-dll-file-in-python
    """
    import ctypes
    from ctypes.wintypes import HANDLE, LPWSTR, DWORD

    GetModuleFileName = ctypes.windll.kernel32.GetModuleFileNameW
    GetModuleFileName.argtypes = HANDLE, LPWSTR, DWORD
    GetModuleFileName.restype = DWORD

    MAX_PATH = 260
    dll = ctypes.CDLL(lib) or ctypes.WINDLL(lib)
    buf = ctypes.create_unicode_buffer(MAX_PATH)
    GetModuleFileName(dll._handle, buf, MAX_PATH)
    abspath = buf.value

    return abspath

def find_library_abspath(lib):
    import os
    if os.name == 'nt':
        return find_library_abspath_windows(lib)
    else:
        return find_library_abspath_linux(lib)
