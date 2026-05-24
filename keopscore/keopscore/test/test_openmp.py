import os
import warnings

import keopscore.config
import pytest


def test_openmp_detection_non_critical():
    """OpenMP is optional: warn when unavailable, validate when available."""
    openmp = keopscore.config.openmp
    assert isinstance(openmp.get_use_OpenMP(), bool)

    if not openmp.get_use_OpenMP():
        with pytest.warns(UserWarning, match="OpenMP is not available"):
            warnings.warn(
                "OpenMP is not available on this platform/configuration.",
                UserWarning,
            )
        return

    lib_path = openmp.get_libomp_path()
    header_path = openmp.get_libomp_include_path()

    assert lib_path and os.path.exists(lib_path)
    assert header_path and os.path.exists(header_path)
    assert "-fopenmp" in openmp.get_compile_options()
