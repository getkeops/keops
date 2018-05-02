import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + (os.path.sep + '..')*2)

import time
import numpy as np

from pykeops.numpy.generic_sum import GenericSum_np


aliases = ["p=Pm(0,1)","a=Vy(1,1)","x=Vx(2,3)","y=Vy(3,3)"]
formula = "Square(p-a)*Exp(x+y)"
signature   =   [ (3, 0), (1, 2), (1, 1), (3, 0), (3, 1) ]
sum_index = 0       # 0 means summation over j, 1 means over i 

p = np.random.randn(1,1)
a = np.random.randn(15000,1)
x = np.random.randn(30000,3)
y = np.random.randn(15000,3)


start = time.time()
c = GenericSum_np("auto",aliases,formula,signature,sum_index,p,a,x,y)

print("time to compute c on cpu : ",round(time.time()-start,2)," seconds")
