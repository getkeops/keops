from c_array import c_array
from c_for import c_for, c_for_loop
from c_function import c_function
from c_if import c_if
from c_toolbox import c_variable, c_block

loop, k = c_for_loop(0, 10, 1, name_incr="k")
code = loop(k.assign(4) + k.add_assign(k))
print(code)

k = c_variable("int", "k")
v = c_array("float", 5, "v")
loop = c_for(
    init=k.declare_assign(0),
    end=k < 10,
    loop=k.plus_plus,
    body=k.assign(4) + k.add_assign(k) + v.assign(0),
)

print(loop)

print(loop.local_vars)
print(loop.global_vars)
