import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.patches as mpatches
from scipy.spatial import distance_matrix
from sklearn.cluster import KMeans
from IPython.display import display, clear_output
import frigidum
from frigidum.examples import tsp

NUM_GU = 10  # number of ground users

X_MIN = 0  # minimum x-axis [meter]
X_MAX = 100  # maximum x-axis [meter]
Y_MIN = 0  # minimum y-axis [meter]
Y_MAX = 100  # maximum y-axis [meter]

UAV_ALTITUDE = 10  # altitude of uav [meter]
MAX_BEAM_ANGLE = 60  # maximum beamforming angle [degree]
UAV_SPEED = 6 # uav speed [m/s]

# calculate maximum beamforming diameter [meter]
MAX_BEAM_DIAMETER = 2 * UAV_ALTITUDE * np.tan(MAX_BEAM_ANGLE * np.pi / 180)
MAX_BEAM_RADIUS = MAX_BEAM_DIAMETER/2

# calculate maximum distance between uav and gu [meter]
MAX_BEAM_DISTANCE = UAV_ALTITUDE / np.cos(MAX_BEAM_ANGLE * np.pi / 180)

X_GRID = 10  # number of x grid
Y_GRID = 10  # number of y grid


TX_POWER = 32  # transmit power [dBm]
F = 1e9  # frequency 1GH [Hz]
C = 299792458  # speed of light [m/s]

UAV_TX_POWER = 30  # uav's transmit power in [dBm]

def calc_rx_power(d):
    # received power [mWh]
    return 1*10**((TX_POWER - (20*np.log10((4*np.pi*d*F)/C)))/10) * 1000
    # generate ground user location randomly x,y,z [meters]

#call information from excel
gu_memory=np.ones((2,NUM_GU))
df=pd.read_excel('locationInformation.xlsx')
for x in range(2):
    for y in range(10):
        gu_memory[x][y]=int(df.iloc[y+1,x])
print(gu_memory)


gu_x = gu_memory[0]

gu_y = gu_memory[1]

gu_z = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

gu_xyz = np.array((gu_x,gu_y,gu_z)).T

gu_xyz
# i use k-mean algorithm 
# to find the center with 7 cluster-head
"""kmeans = KMeans(
    n_clusters=7,
    n_init="auto"
).fit(gu_xyz)

centers = kmeans.cluster_centers_
clear_output(False)
"""
countA=np.zeros(10)
clusterNum=0
RADIUS_FOR_KMEAN=MAX_BEAM_RADIUS #17m(radius) - x(constant)

kmeans = KMeans(
    n_clusters=7,
    n_init="auto"
).fit(gu_xyz)
centers = kmeans.cluster_centers_
clear_output(False)
 

points=np.array([[50,50],
        [34, 37], 
         [89, 20],
         [25.5, 82.5], 
         [2, 9],
         [1, 38],
         [25, 1],
         [51, 19]])

route=[1,0,2,3,4,5,6,7,0]
print(route)
# calculate flying direction vectors
direction = {}
norm_direction = {}
for i in range(len(route) - 1):
    direction[i] = points[route[i+1], :] - points[route[i], :]
    norm_direction[i] = direction[i]/np.linalg.norm(direction[i])

# initial variables
t = 0  # time [second]
index = 0  # index of the route

# initial uav location at time t = 0s
uav_time = {}
uav_time[t] = np.squeeze(points[[route[0],], :])

while True:

    t = t + 1  # increase time t

    # update the next location of uav
    uav_time[t] = np.squeeze(uav_time[t-1] + UAV_SPEED*norm_direction[index])

    # calculate the next location vs. the end point
    tmp = (uav_time[t] - points[route[index+1], :]) / \
        np.linalg.norm(uav_time[t] - points[route[index+1], :])

    # if next location is longer than end point
    # thus, change direction
    if np.all(np.abs(tmp - norm_direction[index]) < 1e-3):
        uav_time[t] = points[route[index+1], :]
        index = index + 1

    # loop until reach the last point
    if index == len(points):
        break

# position of uav at time n

# initial the batter of gus
gu_bat = np.zeros((NUM_GU,))

# call the figure
fig, ax = plt.subplots()
fig.set_figwidth(6)
fig.set_figheight(6)

# design the axis
ax.set_xlabel("x-axis [m]")
ax.set_ylabel("y-axis [m]")
ax.set_xticks(np.arange(X_MIN, X_MAX + 1, X_GRID))
ax.set_yticks(np.arange(Y_MIN, Y_MAX + 1, Y_GRID))
ax.set_xlim(X_MIN, X_MAX)
ax.set_ylim(Y_MIN, Y_MAX)
ax.grid()

# plot initial location of gus with batteries
current_batt = []
for i in range(NUM_GU):
    ax.scatter(x=gu_x[i], y=gu_y[i], c="blue")
    ax.text(x=gu_x[i] - 3.5, y=gu_y[i] - 4, s=f"GU-{i}")
    current_batt.append(
        ax.text(x=gu_x[i] - 6, y=gu_y[i] - 7, s=f"{gu_bat[i]:.2f}mWs"))

#charge station
ax.scatter(x=50,y=50,c="yellow")
ax.text(x=50-5, y=50 - 4, s="Charge")

# plot real time update trajectory
for t in range(len(uav_time) - 1):

    # update title
    ax.set_title(f"Simulation Result [ t = {t}s ]")

    # calculate distance
    distance_uav2gu = distance_matrix(
        [np.append(uav_time[t], UAV_ALTITUDE)], gu_xyz)

    # calculate receive power
    rx_power = calc_rx_power(distance_uav2gu)

    # update the gu battery base on the maximum distance
    gu_bat += (rx_power*(distance_uav2gu <= MAX_BEAM_DISTANCE))[0]

    # plot arrow
    arrow = mpatches.FancyArrowPatch(
        (uav_time[t][0], uav_time[t][1]),
        (uav_time[t+1][0], uav_time[t+1][1]),
        edgecolor="none",
        facecolor="green",
        mutation_scale=20,
        zorder=0
    )
    tmp = ax.add_patch(arrow)

    # scatter uav location
    scatter_uav = ax.scatter(
        x=uav_time[t+1][0],
        y=uav_time[t+1][1],
        c="red",
        marker="s",
        zorder=1
    )

    # remove the previous beam cirle and plot the new one
    if t > 0:
         beam_circle.remove()
         
    beam_circle = Ellipse(
        xy=(uav_time[t+1][0], uav_time[t+1][1]),
        width=MAX_BEAM_DIAMETER,
        height=MAX_BEAM_DIAMETER,
        angle=0,
        edgecolor="none",
        facecolor="orange",
        alpha=0.2,
        zorder=0,
    )
    uav_beam = ax.add_patch(beam_circle)

    # remove the previous battery text
    for i in range(NUM_GU):
        current_batt[i].remove()
        current_batt[i] = ax.text(
            x=gu_x[i] - 6, y=gu_y[i] - 7, s=f"{gu_bat[i]:.2f}mWs")

    # update the figure
    display(fig)
    clear_output(wait=True)

    # set the time sleep
    time.sleep(0.05)