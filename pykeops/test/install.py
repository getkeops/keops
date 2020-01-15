import numpy as np

formula = 'SqNorm2(x - y)'
var = ['x = Vi(3)', 'y = Vj(3)']
expected_res = np.array([63., 90.])

def test_numpy_bindings():
  """
  This function try to compile a simple keops formula using the numpy binder.
  If it fails it turns debug and verbosity flags to True in order to give the 
  user an idea of what is going on...
  """
  x = np.arange(1, 10).reshape(-1, 3).astype('float32')
  y = np.arange(3, 9).reshape(-1, 3).astype('float32')
  
  try:
    import pykeops.numpy as pknp
    my_conv = pknp.Genred(formula, var)
    if np.allclose(my_conv(x, y).flatten(), expected_res):
      print("\npyKeOps with numpy bindings is working!\n")
    else:
      ValueError('[pyKeOps]: outputs wrong values...')
    return

  except:
    import pykeops
    import pykeops.numpy as pknp
    
    pykeops.verbose = True;
    pykeops.build_type = "Debug"
    my_conv = pknp.Genred(formula, var)
    print(my_conv(x, y))

def test_torch_bindings():
  """
  This function try to compile a simple keops formula using the pytorch binder.
  If it fails it turns debug and verbosity flags to True in order to give the 
  user an idea of what is going on...
  """
  try:
    import torch
  except ImportError:
    print("[pyKeops]: torch not found...")  
    return
  except:
    print("[pyKeops]: unexpected error...")  
    return

  x = torch.arange(1, 10, dtype=torch.float32).view(-1, 3)
  y = torch.arange(3, 9, dtype=torch.float32).view(-1, 3)
  
  try:
    import pykeops.torch as pktorch
    my_conv = pktorch.Genred(formula, var)
    if torch.allclose(my_conv(x, y).view(-1), torch.tensor(expected_res).type(torch.float32)):
      print("\npyKeOps with torch bindings is working!\n")
    else:
      ValueError('[pykeOps]: outputs wrong values...')
  except:
    import pykeops
    import pykeops.torch as pktorch
    
    pykeops.verbose = True;
    pykeops.build_type = "Debug"
    my_conv = pktorch.Genred(formula, var)
    print(my_conv(x, y))

