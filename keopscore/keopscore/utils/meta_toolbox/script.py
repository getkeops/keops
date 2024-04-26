from keopscore.utils.meta_toolbox import *

x = c_fixed_size_array("int", 3, "x")
y = c_fixed_size_array("int", 3, "y")
z = c_array_variable("int", "z")

f = lambda out, x, y: f"{out}={x}+{y}"
print(x.apply(f, y, z))
