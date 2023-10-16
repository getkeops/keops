#######################################################################
# .  Warnings, Errors, etc.
#######################################################################

import keopscore
from os.path import join
import re


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
        import subprocess

        out = subprocess.run(command, shell=True, capture_output=True)
        if out.stderr != b"":
            KeOps_Warning("There were warnings or errors :", newline=True)
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


def find_library_abspath(lib):
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


def file_to_string(file_path):
    """
    reads text file and returns its content as a string
    """
    f = open(file_path, "r")
    out = f.read()
    f.close()
    return out


def string_to_file(string, file_path):
    """
    writes string to file
    """
    f = open(file_path, "w")
    out = f.write(string)
    f.close()


def pack_header(filename, origin_folder, target_folder):
    """
    Given a C/C++ header file "filename", located in folder "origin_folder",
    produces a stand-alone version of the header by recursiveley
    replacing all #include "xxx" statements by the content of the corresponding
    file. The resulting file is put into "target_folder".
    """
    code = file_to_string(join(origin_folder, filename))
    used_headers = []
    while True:
        match = re.search('#include *"([^"]*)"', code)
        if match is None:
            break
        header = match.groups()[0]
        if header in used_headers:
            code_to_insert = ""
        else:
            code_to_insert = file_to_string(join(origin_folder, header))
            used_headers.append(header)
        code = code[: match.start()] + code_to_insert + code[match.end() :]
    string_to_file(code, join(target_folder, filename))
