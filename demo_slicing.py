from pykeops.torch import LazyTensor
import torch


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device is:', device)
# Create Vi variable: shape (B=2, M=100, D=10)
data = torch.randn(2, 100, 1, 10).to(device)  # (B, M, 1, D)
D = LazyTensor(data)  # Axis=0 (Vi)

# Slice batch, i, and vector dimensions
D_sliced = D[0, 10:20, :, 3:7]  # New shape: (1, 10, 1, 4)

print('Basic lazytensor shape is:', D.shape)  # Output: (1, 10, 1, 4)
print('Sliced lazytensor shape is:', D_sliced.shape)