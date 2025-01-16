import torch

from pykeops.torch import Genred

x_data = torch.randn(1, 5, 1, 3, device='cpu')

y_data = torch.randn(1, 7, 1, 3, device='cpu')

formula = "Exp(-SqDist(x,y))"

aliases = ["x=Vi(3)", "y=Vj(3)"]

my_op = Genred(formula, aliases=aliases, reduction_op="Sum", axis=1)

result = my_op(x_data, y_data)

print("Expected result shape is [1,5], but result's actual shape is:", result.shape)
