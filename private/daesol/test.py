import numpy as np
import pandas as pd

a=np.ones([2,10])
groupedArea=np.ones(10)
df=pd.read_excel('locationInformation.xlsx')

print(df.iloc[1,0])
for x in range(2):
    for y in range(10):
        a[x][y]=df.iloc[y+1,x]
print(a)
for x in range(10):
    if (abs(a[0][1] - a[0][x])) < 34 and (abs(a[1][1] - a[1][x])) < 34:
        print(x+1+1)


