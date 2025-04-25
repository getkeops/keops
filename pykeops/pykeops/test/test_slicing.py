from pykeops.numpy import LazyTensor
import torch
import numpy as np



def test_slice_keops_vs_torch():
    # Create a simple PyTorch tensor with shape (3, 4, 5)
    x = torch.arange(3 * 4 * 5, dtype=torch.float32).reshape((3, 4, 5))
    # Convert to numpy array to simulate our computed result
    x_np = x
    
    lazy = LazyTensor(x_np)
    
    # In pytorch slicing, this corresponds to: x[1:3, 2, :]
    slice_key = (slice(1, 3), 2, slice(None))
    
    lazy_sliced = lazy[slice_key]
    result_lazy = lazy_sliced()
    
    # Directly slice the original tensor using NumPy slicing
    result_torch = x_np[slice_key]
    
    # Use np.testing.assert_allclose to verify the results match.
    np.testing.assert_allclose(result_lazy, result_torch)
    
    # For visual feedback, print the shapes.
    print("Original shape:", x_np.shape)
    print("KeOps sliced shape:", result_lazy.shape)
    print("PyTorch sliced shape:", result_torch.shape)