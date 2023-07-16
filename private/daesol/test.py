import numpy as np
import pandas as pd

groupArea=[4,5,8,11,11]
tmp=[0,2,3,8,11]
x=np.unique(groupArea)
y=np.unique(tmp)
z=np.zeros((10,2))
z=[[2, 2, 4, 4, 3, 2, 1, 0, 0],
   [4, 3, 2, 1, 2, 3, 4, 4, 0]]

#print(np.unique(np.union1d(x,y)))
print(z[0].count(int(4)))



