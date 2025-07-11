import importlib.util


def find_package_location(package_name: str) -> str:
    """Find the __init__ file of a given package

    This function does not import the package, it was written to avoid circular
    imports with KeOps

    Parameters
    ----------
    package_name
        The name of the package

    Returns
    -------
    str
        The path to the package

    Raises
    ------
    ImportError
        If the package cannot be loaded

    """
    spec = importlib.util.find_spec(package_name)
    if spec.origin:
        return spec.origin
    else:
        message = f"Package '{package_name}' not found."
        raise ImportError(message)
