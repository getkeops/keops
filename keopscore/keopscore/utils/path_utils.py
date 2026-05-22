import os
import site
import sys
import sysconfig
from pathlib import Path


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


def _ordered_search_roots(
    env_vars=(), pip_suffixes=(), conda_root=None, system_roots=(), order=None
):
    """Return normalized search roots in a caller-defined category order.

    Args:
        env_vars (tuple[str] | list[str]): Environment variable names whose values
            should be considered as root directories.
        pip_suffixes (tuple[str] | list[str]): Relative suffixes appended to Python
            package roots discovered from the current interpreter.
        conda_root (str | None): Name of an environment variable that points to a
            Conda root directory.
        system_roots (tuple[str] | list[str]): Fallback root directories to append.
        order (tuple[str] | list[str] | str | None): Ordered categories to apply.
            Allowed items are ``env_vars``, ``pip_suffixes``, ``conda_root``, and
            ``system_roots``. A comma-separated string is also accepted.

    Returns:
        list[pathlib.Path]: Existing candidates, deduplicated in first-seen order.
    """

    if order is None:
        order = ("env_vars", "conda_root", "system_roots", "pip_suffixes")
    else:
        isinstance(order, str) and (order := tuple(order.split(",")))
        for item in order:
            if item not in (
                "env_vars",
                "pip_suffixes",
                "conda_root",
                "system_roots",
            ):
                raise ValueError(f"Invalid order item: {item}")

    roots = []

    for item in order:
        if item == "env_vars" and env_vars:
            roots.extend(_env_roots(env_vars))

        if item == "pip_suffixes" and pip_suffixes:
            # Pip wheels such as nvidia-cuda-runtime expose libraries under site-packages.
            roots.extend(_path_candidates(_python_package_roots(), pip_suffixes))

        if item == "conda_root" and conda_root:
            roots.extend(_env_roots((conda_root,)))

        if item == "system_roots" and system_roots:
            roots.extend(system_roots)

    return _unique_paths(roots)


def _first_matching_file(directories, patterns):
    """Return the first file matching glob patterns inside ordered directories."""
    if not directories or not patterns:
        return None

    if isinstance(patterns, str):
        patterns = (patterns,)
    patterns = [pattern for pattern in patterns if pattern]
    if not patterns:
        return None

    for directory in _unique_paths(directories):
        try:
            directory = Path(directory)
            if not directory.is_dir():
                continue
            for pattern in patterns:
                for match in sorted(directory.glob(pattern)):
                    if match.is_file():
                        return str(match)
        except Exception:
            continue

    return None


def ensure_directory(path, add_to_syspath=False):
    """Create a directory and optionally register it in sys.path."""
    os.makedirs(path, exist_ok=True)
    if add_to_syspath and path not in sys.path:
        sys.path.append(path)
    return path
