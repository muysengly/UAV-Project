import numpy as np
X_MIN = 0  # minimum x-axis [meter]
X_MAX = 100  # maximum x-axis [meter]
Y_MIN = 0  # minimum y-axis [meter]
Y_MAX = 100  # maximum y-axis [mseter]
from scipy.spatial import distance_matrix
TX_POWER = 32
NUM_GU = 10
F = 1e9  # frequency 1GH [Hz]
C = 299792458
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
def calc_minhovtime(i):
    distance_uav2gu = np.squeeze(distance_matrix(
                [np.append(centers[i], UAV_ALTITUDE)], gu_xyz))
    print(distance_uav2gu)
    print(distance_uav2gu[np.argmin(distance_uav2gu)])
    return 100/calc_rx_power(distance_uav2gu[np.argmin(distance_uav2gu)])

def calc_rx_power(d):
    # received power [mWh]
    return 1*10**((TX_POWER - (20*np.log10((4*np.pi*d*F)/C)))/10) * 1000

centers=[[50, 50],
         [1.5, 23.5],
         [35.5, 42.5],
         [89, 20],
         [35.6667, 15.33],
         [25.5, 82.5]]
center2=[[89, 20],
         [35.6667, 15.33],
         [25.5, 82.5]]
centers.append(center2)
print(centers)