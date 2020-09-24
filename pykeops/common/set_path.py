import os
import sys
from hashlib import sha256

from pykeops import __version__ as version


def set_bin_folder(bf_user=None, append_to_python_path=True):
    """
    This function set a default build folder that contains the python module compiled
    by pykeops.
    """
    bf_source = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'build')
    bf_home = os.path.expanduser('~')
    name = 'pykeops-{}-{}'.format(version, sys.implementation.cache_tag)
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        name += "-gpu" + os.environ['CUDA_VISIBLE_DEVICES'].replace(',', '-')

    if bf_user is not None:  # user provide an explicit path
        bin_folder = os.path.expanduser(bf_user)
    elif os.path.isdir(bf_source):  # assume we are loading from source
        bin_folder = bf_source
    elif os.path.isdir(bf_home):  # assume we are using wheel and home is accessible
        bin_folder = os.path.join(bf_home, '.cache', name)
    else:
        import tempfile
        bin_folder = tempfile.mkdtemp(prefix=name)

    # Clean path name
    bin_folder = os.path.realpath(bin_folder)
    if not bin_folder.endswith(os.path.sep):
        bin_folder += os.path.sep

    # Save the path and append in python path
    if append_to_python_path:
        sys.path.append(bin_folder)

    return bin_folder


def create_name(formula, aliases, dtype, lang, optional_flags):
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


def set_build_folder(bin_folder, dll_name):
    return os.path.join(bin_folder, 'build-' + dll_name)