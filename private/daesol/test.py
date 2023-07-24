import numpy as np
import pandas as pd

groupArea=[4,5,8,11,11]
tmp=[0,2,3,8,11]
x=np.unique(groupArea)
y=np.unique(tmp)
z=np.zeros((10,2))
z=[[2, 2, 4, 4, 3, 2, 1, 0, 0],
   [4, 3, 2, 1, 2, 3, 4, 4, 0]]
route = [0, 2,3,2, 4]

center=[[20, 60, 75],
        [30, 45, 95]]
centers=[[50, 50],
         [1.5, 23.5],
         [35.5, 42.5],
         [89, 20],
         [35.6667, 15.33],
         [25.5, 82.5]]
t=[[30, 35.5, 90],
   [10, 42.5, 99]]

#print(np.unique(np.union1d(x,y))) 
for a in range(1,6):
   for i in range(3):
      if np.any(np.in1d(centers[a][:0],t[0][i]))==1 and np.any(np.in1d(centers[a,:1],t[1][i]))==1:
         print("aa")
      else:
         print("bb")
   

"""if np.any(np.in1d(route,2))==1:
    print("ss")
else:
    print("zz")
"""

