import numpy as np
X_MIN = 0  # minimum x-axis [meter]
X_MAX = 100  # maximum x-axis [meter]
Y_MIN = 0  # minimum y-axis [meter]
Y_MAX = 100  # maximum y-axis [mseter]
from scipy.spatial import distance_matrix

NUM_GU = 10
UAV_ALTITUDE = 10
MAX_BEAM_ANGLE = 60  # maximum beamforming angle [degree]
# maximum beamforming diameter [meter]
MAX_BEAM_DIAMETER = 2*UAV_ALTITUDE*np.tan(MAX_BEAM_ANGLE*np.pi/180)
MAX_BEAM_RADIUS = MAX_BEAM_DIAMETER/2
MAX_BEAM_DISTANCE = UAV_ALTITUDE / np.cos(MAX_BEAM_ANGLE * np.pi / 180)
gu_x = np.random.uniform(low=X_MIN, high=X_MAX, size=(NUM_GU,))
gu_y = np.random.uniform(low=Y_MIN, high=Y_MAX, size=(NUM_GU,))
gu_x = np.array([10., 20., 30., 56., 24.,12., 67., 55., 94., 2.])
gu_z = np.array([23., 75., 68., 55., 45., 23., 44., 87., 65., 13.])
#gu_z = np.zeros((NUM_GU,))
gu_z = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
gu_xyz = np.array((gu_x,gu_y,gu_z)).T

centers=[[50, 50],
         [1.5, 23.5],
         [35.5, 42.5],
         [89, 20],
         [35.6667, 15.33],
         [25.5, 82.5]]
for i in range(2):
    for x in range(5):
        distance_uav2gu = np.squeeze(distance_matrix(
                            [np.append(centers[x], UAV_ALTITUDE)], gu_xyz))
        distance_center=np.squeeze(distance_uav2gu <= MAX_BEAM_DISTANCE)
        print(distance_uav2gu)
