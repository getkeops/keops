
from keopscore.utils.code_gen_utils import c_array, c_variable, c_tensor, VectApply

x=c_array("int",3,"x")
y=c_array("int",1,"y")
z=c_array("int",3,"z")

y=c_variable("int","y")

x=c_tensor("int",(3,2),"x")
y=c_tensor("int",(1,2),"y")
z=c_tensor("int",(3,2),"z")

plus = lambda x,y,z : f"{x.id} = {y.id}+{z.id};\n"

code = VectApply(plus, x, y, z)

print(code)