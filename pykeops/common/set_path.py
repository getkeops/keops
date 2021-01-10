import os
import shutil
import sys
import warnings
from hashlib import sha256

from pykeops import __version__ as version
import pykeops.config


def set_bin_folder(bf_user=None, append_to_python_path=True):
    """
    This function set a default build folder that contains the python module compiled
    by pykeops. It populates the pykeops.config.build_folder variable (str) and
    add the folder to the python path.
    """
    bf_source = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build")
    bf_home = os.path.expanduser("~")
    name = "pykeops-{}-{}".format(version, sys.implementation.cache_tag)
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        name += "-gpu" + os.environ["CUDA_VISIBLE_DEVICES"].replace(",", "-")

    if bf_user is not None:  # user provide an explicit path
        bin_folder = os.path.expanduser(bf_user)
    elif os.path.isdir(bf_source):  # assume we are loading from source
        bin_folder = bf_source
    elif os.path.isdir(bf_home):  # assume we are using wheel and home is accessible
        bin_folder = os.path.join(bf_home, ".cache", name)
    else:
        import tempfile

        bin_folder = tempfile.mkdtemp(prefix=name)

    # Clean path name
    bin_folder = os.path.realpath(bin_folder)
    if not bin_folder.endswith(os.path.sep):
        bin_folder += os.path.sep

    # Create the bin_folder dir here... as importing a non existing dir makes python not happy...
    os.makedirs(bin_folder, exist_ok=True)

    # Save the path and append in python path
    if append_to_python_path:
        while pykeops.config.bin_folder in sys.path:
            sys.path.remove(pykeops.config.bin_folder)
        sys.path.append(bin_folder)
        if any(
            any(i in s for i in ["libKeOps", "fshape_scp", "radial_kernel"])
            for s in sys.modules.keys()
        ):  # check that no pykeops modules were imported.
            warnings.warn(
                "[pyKeOps]: set_bin_folder() has been invoked while some pykeops modules have already "
                "been imported. To avoid this warning, change pykeops.config.bin_folder just after importing"
                " pykeops, that is, before any computations."
            )

    pykeops.config.bin_folder = bin_folder


def create_name(formula, aliases, dtype, lang, optional_flags):
    """
    Compose the shared object name
    """
    formula = formula.replace(" ", "")  # Remove spaces
    aliases = [alias.replace(" ", "") for alias in aliases]

    # Since the OS prevents us from using arbitrary long file names, an okayish solution is to call
    # a standard hash function, and hope that we won't fall into a non-injective nightmare case...
    dll_name = ",".join(aliases + [formula] + optional_flags) + "_" + dtype
    dll_name = (
        "libKeOps"
        + lang
        + sha256((pykeops.config.bin_folder + dll_name).encode("utf-8")).hexdigest()[
            :10
        ]
    )
    return dll_name


def set_build_folder(bin_folder, dll_name):
    return os.path.join(bin_folder, "build-" + dll_name)


def clean_pykeops(path="", lang=""):
    if lang not in ["numpy", "torch", ""]:
        raise ValueError(
            '[pyKeOps:] lang should be the empty string, "numpy" or "torch"'
        )

    if path == "":
        path = pykeops.config.bin_folder

    print("Cleaning " + path + "...")

    for f in os.scandir(path):
        if f.is_dir(follow_symlinks=False) and (
            f.name.count("build-" + lang)
            or f.name.count("build-pybind11_template-libKeOps")
            or f.name.count("libKeOps" + lang)
            or f.name.count("build-fshape_scp")
            or f.name.count("build-radial_kernel")
        ):
            shutil.rmtree(f.path)

        elif f.is_file() and (
            f.name.count("fshape_scp")
            or f.name.count("radial_kernel")
            or f.name.count("keops_hash")
        ):
            os.remove(f.path)
        else:
            continue

        print("    - " + f.path + " has been removed.")
