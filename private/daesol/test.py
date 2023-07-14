import numpy as np
import pandas as pd

groupArea=[4,5,8,11,11]
tmp=[0,2,3,8,11]
x=np.unique(groupArea)
y=np.unique(tmp)
#print(np.unique(np.union1d(x,y)))
print(x)
print(y)
for a in range(len(x)):
    print(len(np.intersect1d(x[a],y)))



