import numpy as np
import pandas as pd

groupArea=[6,1,1,1,1]
tmp=[8,1,1,1,1]
x=np.unique(groupArea)
y=np.unique(tmp)
print(np.unique(np.union1d(x,y)))
if np.array_equal(groupArea,tmp)==0:
    print("case3")
    print(groupArea)


