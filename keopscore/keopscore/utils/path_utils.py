import os
import sys
from pathlib import Path


def _unique_paths(paths):
    """Return existing paths in first-seen order with duplicates and None values removed."""
    unique = []
    seen = set()
    for path in paths:
        if path is None:
            continue
        path = Path(path)
        if not path.exists() and not path.is_dir():
            continue
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


def _ordered_search_roots(env_vars=(), pip=(), conda=None, system=(), order=None):
    """Return normalized search roots in a caller-defined category order.

    Args:
        env_vars (tuple[str] | list[str]): Environment variable names whose values
            should be considered as root directories.
        pip (tuple[str] | list[str]): Relative suffixes appended to Python
            package roots discovered from the current interpreter.
        conda (str | None): Name of an environment variable that points to a
            Conda root directory.
        system (tuple[str] | list[str]): Fallback root directories to append.
        order (tuple[str] | list[str] | str | None): Ordered categories to apply.
            Allowed items are ``env_vars``, ``pip``, ``conda``, and
            ``system``. A comma-separated string is also accepted.

    Returns:
        list[pathlib.Path]: Existing candidates, deduplicated in first-seen order.
    """

    if order is None:
        order = ("env_vars", "conda", "system", "pip")
    else:
        isinstance(order, str) and (order := tuple(order.split(",")))
        for item in order:
            if item not in (
                "env_vars",
                "pip",
                "conda",
                "system",
            ):
                raise ValueError(f"Invalid order item: {item}")

    roots = []

    for item in order:
        if item == "env_vars" and env_vars:
            roots.extend(_env_roots(env_vars))

        if item == "pip" and pip:
            roots.extend(_unique_paths(pip))

        if item == "conda" and conda:
            roots.extend(_env_roots((conda,)))

        if item == "system" and system:
            roots.extend(_unique_paths(system))

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
