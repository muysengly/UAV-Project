from scipy.spatial import distance_matrix
import numpy as np
import pandas as pd
UAV_ALTITUDE=10
NUM_GU=10
MAX_BEAM_ANGLE = 60  # maximum beam-forming angle [degree]
MAX_BEAM_DISTANCE = UAV_ALTITUDE / np.cos(MAX_BEAM_ANGLE * np.pi / 180)
centers=[[50, 50],
         [1.5, 23.5],
         [35.5, 42.5],
         [89, 20],
         [35.6667, 15.33],
         [25.5, 82.5]]

gu_memory=np.ones((2,NUM_GU))
df=pd.read_excel('locationInformation.xlsx')
for x in range(2):
    for y in range(10):
        gu_memory[x][y]=int(df.iloc[y+1,x])
gu_x = gu_memory[0]
gu_y = gu_memory[1]
gu_z = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
gu_xyz = np.array((gu_x,gu_y,gu_z)).T

distance_uav2gu = distance_matrix(
            [np.append(centers[2], UAV_ALTITUDE)], gu_xyz)
distance_center=np.squeeze(distance_uav2gu <= MAX_BEAM_DISTANCE)
distance_index=np.where(distance_center == True)
print(distance_center)
print(distance_index[0])
print(distance_uav2gu)
#100/calcrxpower