# python -m pybind11 --cmakedir
# C:\ProgramData\Anaconda3\envs\pt1.11_cu11.3\lib\site-packages\pybind11\share\cmake\pybind11
# python -m pybind11 --includes
# -IC:\ProgramData\Anaconda3\envs\pt1.11_cu11.3\Include -IC:\ProgramData\Anaconda3\envs\pt1.11_cu11.3\lib\site-packages\pybind11\include

from keopscore.config.config import get_compiler_library
print('libcuda =', get_compiler_library('libcuda'))
print('libnvrtc =', get_compiler_library('libnvrtc'))
print('libcudart =', get_compiler_library('libcudart'))

import keopscore
import pykeops


pykeops.test_numpy_bindings()
pykeops.test_torch_bindings()

pass
