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
    env_vars=(), pip_suffixes=(), conda_root=None, system_roots=()
):
    """Return roots ordered by explicit env vars, pip, conda, then system paths."""
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
