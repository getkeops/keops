## conda install -c conda-forge portalocker pywin32
## pip install GPUtil
#%%
import os
os.environ['PYKEOPS_VERBOSE'] = "1"
os.environ['CC'] = "clang-cl.exe"
os.environ['CXX'] = "clang-cl.exe"
import numpy as np
import pykeops.numpy as pknp


x = np.arange(1, 10).reshape(-1, 3).astype('float32')
y = np.arange(3, 9).reshape(-1, 3).astype('float32')

my_conv = pknp.Genred('SqNorm2(x - y)', ['x = Vi(3)', 'y = Vj(3)'])
print(my_conv(x, y))

#%%
import os
os.environ['PYKEOPS_VERBOSE'] = "1"
os.environ['CC'] = "clang-cl.exe"
os.environ['CXX'] = "clang-cl.exe"
import torch
import pykeops.torch as pktorch

x = torch.arange(1, 10, dtype=torch.float32).view(-1, 3)
y = torch.arange(3, 9, dtype=torch.float32).view(-1, 3)

my_conv = pktorch.Genred('SqNorm2(x-y)', ['x = Vi(3)', 'y = Vj(3)'])
print(my_conv(x, y)) 

#%%
import os
os.environ['PYKEOPS_VERBOSE'] = "1"
os.environ['CC'] = "clang-cl.exe"
os.environ['CXX'] = "clang-cl.exe"
import subprocess
build_folder="c:\\Users\\Franc\\Documents\\dev\\python\\keops\\pykeops\\build\\build-dir"
# args = ['cmake', '--build', '.', '--target', 'libKeOpsnumpy5ac3d464a2']
args = ['cmake', 'c:\\Users\\Franc\\Documents\\dev\\python\\keops\\pykeops', '-GNinja', '-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON', '-DCMAKE_BUILD_TYPE=Release', '-DFORMULA_OBJ=Sum_Reduction(SqNorm2(x - y),1)', '-DVAR_ALIASES=auto x = Vi(0,3); auto y = Vj(1,3); ', '-Dshared_obj_name=libKeOpsnumpy5ac3d464a2', '-D__TYPE__=double', '-DPYTHON_LANG=numpy', '-DC_CONTIGUOUS=1']
try:
    proc = subprocess.run(args, cwd=build_folder, shell=True,stdout=subprocess.PIPE, check=True)
    print(proc.stdout.decode('utf-8'))
except subprocess.CalledProcessError as e:
    print(e)
    print(e.stdout.decode('cp850'))
#%%
' '.join(args)

#%%
