import numpy as np
import torch


def assert_torch_allclose(actual, expected, *, label=None, **kwargs):
    # test dtype, convert long to float if needed

    if actual.dtype in [torch.int64, torch.int32]:
        actual = actual.float()

    if expected.dtype in [torch.int64, torch.int32]:
        expected = expected.float()

    ok = torch.allclose(actual, expected, **kwargs)
    diff = torch.linalg.norm(actual - expected).item()
    prefix = label if label is not None else "torch.allclose failed"
    assert ok, f"{prefix}: ||ref-test||_2={diff:.6e} and ||ref||_2={torch.linalg.norm(expected).item():.6e}"


def assert_np_allclose(actual, expected, *, label=None, **kwargs):
    ok = np.allclose(actual, expected, **kwargs)
    diff = float(np.linalg.norm(np.asarray(actual) - np.asarray(expected)))
    prefix = label if label is not None else "np.allclose failed"
    assert ok, f"{prefix}: ||ref-test||_2={diff:.6e} and ||ref||_2={float(np.linalg.norm(expected)):.6e}"
