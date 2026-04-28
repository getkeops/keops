import os
import site
import sys
import sysconfig
from pathlib import Path

from keopscore.utils.misc_utils import CHECK_MARK, CROSS_MARK, find_library_abspath

not_found_str = f"Not Found. {CROSS_MARK}"
enabled_dict = {True: f"Enabled {CHECK_MARK}", False: f"Disabled {CROSS_MARK}"}

def _unique_paths(paths):
    """Return paths in first-seen order with duplicates and None values removed."""
    unique = []
    seen = set()
    for path in paths:
        if path is None:
            continue
        path = Path(path)
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _env_roots(env_vars):
    """Return existing environment variable values as normalized Path objects."""
    return _unique_paths(os.getenv(env_var) for env_var in env_vars)


def _path_candidates(roots, suffixes):
    """Expand root directories with relative suffixes while preserving order."""
    candidates = []
    for root in _unique_paths(roots):
        for suffix in suffixes:
            candidates.append(root / suffix if suffix else root)
    return _unique_paths(candidates)


def _first_existing_file(paths):
    """Return the first path that points to an existing file."""
    for path in _unique_paths(paths):
        if path.is_file():
            return str(path)
    return None


def _first_matching_file(directories, patterns):
    """Return the first file matching glob patterns inside ordered directories."""
    for directory in _unique_paths(directories):
        if not directory.is_dir():
            continue
        for pattern in patterns:
            matches = sorted(directory.glob(pattern))
            for match in matches:
                if match.is_file():
                    return str(match)
    return None


def _first_existing_dir_with_files(directories, required_filenames):
    """Return the first directory containing all required filenames."""
    for directory in _unique_paths(directories):
        if all((directory / filename).is_file() for filename in required_filenames):
            return str(directory)
    return None


def _find_library_by_names(library_names):
    """Return the first library path resolved by ctypes for known library names."""
    from ctypes.util import find_library

    # ctypes delegates to the platform loader, so this is the final fallback.
    for library_name in library_names:
        library_path = find_library(library_name)
        if library_path:
            return find_library_abspath(library_name)
        """
        library_path = find_library(library_name)

        if library_path:    
            full_path = KeOps_OS_Run("ldconfig -p").stdout.decode("utf-8")

            #print(f"Checking for OpenMP library '{full_path}' in ldconfig output...")
            for line in full_path.splitlines():
                if library_path in line:
                    library_full_path = line.split("=>", 1)[1].strip()
                    return library_path, os.path.dirname(library_full_path)
        """

    return None


def _python_package_roots():
    """Return Python package roots where pip-installed wheels may live."""
    package_roots = []
    getters = [
        lambda: getattr(site, "getsitepackages", lambda: [])(),
        lambda: [site.getusersitepackages()],
        lambda: [sysconfig.get_path("purelib")],
        lambda: [sysconfig.get_path("platlib")],
        lambda: sys.path,
    ]
    for getter in getters:
        try:
            # Some site helpers are unavailable in embedded or non-standard Python builds.
            package_roots.extend(getter() or [])
        except Exception:
            continue
    return _unique_paths(package_roots)


def _ordered_search_roots(env_vars=(), pip_suffixes=(), conda_root=None, system_roots=()):
    """Return roots ordered by explicit env vars, pip, conda, then system paths.

    :argument
        - env_vars (list): environment variable names to search for paths
        - pip_suffixes (list): suffixes to append to pip-installed wheel paths
        - conda_root (str): environment variable name for conda root
        - system_roots (list): explicit system paths to search
    """
    roots = []

    if env_vars:
        roots.extend(_env_roots(env_vars))

    if pip_suffixes:
        # Pip wheels such as nvidia-cuda-runtime expose libraries under site-packages.
        roots.extend(_path_candidates(_python_package_roots(), pip_suffixes))

    if conda_root:
        roots.extend(_env_roots((conda_root,)))
    roots.extend(system_roots)
    return _unique_paths(roots)


def ensure_directory(path, add_to_syspath=False):
    """Create a directory and optionally register it in sys.path."""
    os.makedirs(path, exist_ok=True)
    if add_to_syspath and path not in sys.path:
        sys.path.append(path)
    return path


def print_envs(env_vars):
    """Print the values of specified environment variables."""
    print("\nRelevant Environment Variables")
    print("-" * 60)
    for var in env_vars:
        value = os.environ.get(var)
        if value:
            print(f"{var} = {value}")
        else:
            print(f"{var} is not set")