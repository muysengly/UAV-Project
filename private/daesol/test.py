import numpy as np
import pandas as pd

qtable=np.zeros((7,6,4))

#read centers
print(pd.read_csv('centers.csv'))
df=pd.read_csv('centers.csv')
df = df.drop(df.columns[0], axis=1)
centers=df.to_numpy()
print(centers)

#read routes
print(pd.read_csv('routes.csv'))
df=pd.read_csv('routes.csv')
df = df.drop(df.columns[0], axis=1)
route=df.to_numpy()
route=np.squeeze(route.T)
print(route)