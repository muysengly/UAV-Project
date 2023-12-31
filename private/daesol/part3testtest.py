import time
import math
import frigidum
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, clear_output
from frigidum.examples import tsp
from matplotlib.patches import Ellipse
from scipy.spatial import distance_matrix
from sklearn.cluster import KMeans
import pandas as pd

def makeUAV(xpos,ypos,maxbeam,uavno):
    uav_number = "UAV"+str(uavno)
    plt.scatter(x=xpos, y=ypos, c="green")
    plt.text(x=xpos - 3.5, y=ypos - 4, s=uav_number)

def calc_rx_power(d):
    # received power [mWh]
    return 1*10**((TX_POWER - (20*np.log10((4*np.pi*d*F)/C)))/10) * 1000

def distance3d(center,i,j):
    return (((center[i][0]-center[j][0])**2)+((center[i][1]-center[j][1])**2)+(UAV_ALTITUDE**2))**(1/2)

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (UAV_ALTITUDE**2))

def tsp_dp(points):
    n = len(points)
    all_bits = (1 << n) - 1

    memo = [[-1] * n for _ in range(1 << n)]

    def solve(mask, pos):
        if mask == all_bits:
            return euclidean_distance(points[pos], points[0]), [0]

        if memo[mask][pos] != -1:
            return memo[mask][pos]

        min_dist = float('inf')
        best_route = []
        for next_pos in range(n):
            if (mask & (1 << next_pos)) == 0:
                new_dist, route = solve(mask | (1 << next_pos), next_pos)
                new_dist += euclidean_distance(points[pos], points[next_pos])
                if new_dist < min_dist:
                    min_dist = new_dist
                    best_route = [next_pos] + route

        memo[mask][pos] = min_dist, best_route
        return min_dist, best_route

    shortest_distance, route = solve(1, 0)
    route = [0] + route  # Add the starting city to the route
    return shortest_distance, route

def calc_totaltime(finald):
    totaltime=0
    hovtime=0
    movtime=finald/UAV_SPEED 
    for calchov in range(len(centers)):
        time1=[]
        distance_uav2gu = np.squeeze(distance_matrix(
                    [np.append(centers[calchov], UAV_ALTITUDE)], gu_xyz))
        distance_center=np.squeeze(distance_uav2gu <= MAX_BEAM_DISTANCE)
        distance_index=np.where(distance_center == True)
        for charge in range(len(distance_index[0])):
            time1=np.append(time1,100/calc_rx_power(distance_uav2gu[distance_index[0][charge]]))
        if len(time1)==1:
            hovtime+=time1[0]
        if len(time1)>1:
            hovtime+=np.max(time1)
    totaltime+=(hovtime+movtime)
    return round(totaltime,3)

def calc_hovtime():
    hovtime=0
    for calchov in range(len(centers)):
        time1=[]
        distance_uav2gu = np.squeeze(distance_matrix(
                    [np.append(centers[calchov], UAV_ALTITUDE)], gu_xyz))
        distance_center=np.squeeze(distance_uav2gu <= MAX_BEAM_DISTANCE)
        distance_index=np.where(distance_center == True)
        for charge in range(len(distance_index[0])):
            time1=np.append(time1,100/calc_rx_power(distance_uav2gu[distance_index[0][charge]]))
        if len(time1)==1:
            hovtime+=time1[0]
        if len(time1)>1:
            hovtime+=np.max(time1)
    return hovtime

def calc_movtime(finald):
    return finald/UAV_SPEED

def calc_usedpower(finald):
    usedpower=(NUM_GU*GU_MAXBATT)+(calc_hovtime()*UAV_HOV)+(calc_movtime(finald)*UAV_MOVE)
    return round(usedpower,3)

#intial parameter
NUM_GU = 10  # number of ground users
TX_POWER = 32  # transmit power [dBm]
F = 1e9  # frequency 1GH [Hz]
C = 299792458  # speed of light [m/s]
UAV_TX_POWER = 30  # uav's transmit power in [dBm]
X_MIN = 0  # minimum x-axis [meter]
X_MAX = 100  # maximum x-axis [meter]
Y_MIN = 0  # minimum y-axis [meter]
Y_MAX = 100  # maximum y-axis [mseter]
UAV_ALTITUDE = 10  # altitude of uav [meter]
MAX_BEAM_ANGLE = 60  # maximum beamforming angle [degree]
# maximum beamforming diameter [meter]
MAX_BEAM_DIAMETER = 2*UAV_ALTITUDE*np.tan(MAX_BEAM_ANGLE*np.pi/180)
MAX_BEAM_RADIUS = MAX_BEAM_DIAMETER/2
MAX_BEAM_DISTANCE = UAV_ALTITUDE / np.cos(MAX_BEAM_ANGLE * np.pi / 180)
X_GRID = 10  # number of x grid
Y_GRID = 10  # number of y grid
UAV_TX_POWER = 30  # uav's transmit power in [dBm]

MAX_UAV_BATTERY=800 #uav's battery; 500mW
UAV_SPEED = 6 #[m/s]
UAV_MOVE= 5 #10mWs when moving
UAV_HOV= 2 # 5mWs when hovering
UAV_LOWBATT = 250 # go to charge station when UAV's battery left is 200mW
GU_MAXBATT=100

#initial variables
t = 0  # time [seconds]
gu_bat = np.zeros((NUM_GU,)) # battery of ground user [mWh]
# generate meshgrid
tmp_x = np.linspace(X_MIN + X_GRID/2,X_MAX-X_GRID/2,X_GRID)
tmp_y = np.linspace(Y_MIN + Y_GRID/2,Y_MAX-Y_GRID/2,Y_GRID)
GRID = np.array(np.meshgrid(tmp_x, tmp_y))
GRID
color=["red","orange","yellow","green","olive","blue","skyblue","violet","brown","pink"]

fig, ax = plt.subplots(figsize=(6, 6))


#call GU
gu_memory=np.ones((2,NUM_GU))
file_path = '/content/drive/MyDrive/locationInformation.xlsx'
df = pd.read_excel('locationInformation.xlsx')
for x in range(2):
    for y in range(10):
        gu_memory[x][y]=int(df.iloc[y+1,x])
gu_x = gu_memory[0]
gu_y = gu_memory[1]
gu_z = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
gu_xyz = np.array((gu_x,gu_y,gu_z)).T

current_batt = []
for i in range(NUM_GU):
    ax.scatter(x=gu_x[i], y=gu_y[i], c="blue")
    ax.text(x=gu_x[i] - 3.5, y=gu_y[i] - 4, s=f"GU-{i}")
    current_batt.append(
        ax.text(x=gu_x[i] - 6, y=gu_y[i] - 7, s=f"{gu_bat[i]}mWh"))


centers=[[50, 50],
         [1.5, 23.5],
         [35.5, 42.5],
         [89, 20],
         [35.6667, 15.33],
         [25.5, 82.5]]
"""print(pd.read_csv('centers.csv'))
df=pd.read_csv('centers.csv')
df = df.drop(df.columns[0], axis=1)
centers=df.to_numpy()
print(centers)"""

for i in range(len(centers)):
    plt.scatter(x=centers[i][0], y=centers[i][1], c=color[9])
    plt.text(x=centers[i][0] - 1.5, y=centers[i][1] + 2, s=f"C-{i}")

#TSP algorithm
#used Dynamic Programming

#input: centers: cluster centers array(x,y)
#output1: route array

#output2: time taken(=hovering time + moving time)
#moving time= distance / uav_speed
#hovering time=sum of time taken to charge all GU's per cluster

#output3: used power(=(100mW * NUM_GU) + (hovering time * 2mW) + (moving time *5mW) )


start_time = time.time()
shortest_distance, route = tsp_dp(centers)
end_time = time.time()



print("Shortest distance:", shortest_distance)
print("Time taken:(program)", end_time - start_time, "seconds")
print("Route:", route)
print(f"total time taken={calc_totaltime(shortest_distance)}s, used power={calc_usedpower(shortest_distance)}mW")

#drawing
for i in range(len(route)-1):
    x_vals = [centers[int(route[i])][0], centers[int(route[i + 1])][0]]
    y_vals = [centers[int(route[i])][1], centers[int(route[i + 1])][1]]
    plt.plot(x_vals, y_vals, color=color[i])

# Re-clustering 알고리즘
def recluster_gus(current_cluster_centers, current_gus, max_gus_per_cluster=2):
    new_cluster_centers = []
    new_gus = []

    for center_idx, center in enumerate(current_cluster_centers):
        # 현재 cluster에 포함된 GU 찾기
        gu_indices = [i for i, gu in enumerate(current_gus) if gu['cluster'] == center_idx]
        gu_positions = np.array([(gu['x'], gu['y']) for idx, gu in enumerate(current_gus) if idx in gu_indices])

        if len(gu_indices) > max_gus_per_cluster:
            # k-평균 알고리즘을 사용하여 새로운 cluster center 계산
            kmeans = KMeans(n_clusters=max_gus_per_cluster, n_init="auto").fit(gu_positions)
            new_centers = kmeans.cluster_centers_

            # 새로운 cluster에 GU 배정
            new_gu_indices = kmeans.predict(gu_positions)

            for idx, new_center in enumerate(new_centers):
                new_cluster_centers.append(new_center)
                new_gus.append({'x': new_center[0], 'y': new_center[1], 'cluster': center_idx, 'battery': GU_MAXBATT})

            # 새 cluster의 일부분이 아닌 GU 위치 업데이트 
            remaining_gu_indices = [i for i in gu_indices if i not in np.where(new_gu_indices == idx)[0]]
            for idx in remaining_gu_indices:
                new_gus.append(current_gus[idx])
        else:
            # 아니면 동일한 cluster 중심과 CU 위치 유지
            new_cluster_centers.append(center)
            for idx in gu_indices:
                new_gus.append(current_gus[idx])

    return new_cluster_centers, new_gus

# 초기 GU 정보로 current_gus 초기화
current_gus = [{'x': gu_x[i], 'y': gu_y[i], 'cluster': -1, 'battery': gu_bat[i]} for i in range(NUM_GU)]

# re-clustering loop
for center_idx in range(len(route) - 1):
    current_center = centers[route[center_idx]]
    next_center = centers[route[center_idx + 1]]

    # UAV를 다음 센터로 이동
    distance_to_next_center = euclidean_distance(current_center, next_center)
    time_to_move = distance_to_next_center / UAV_SPEED
    current_uav_position = (current_center[0], current_center[1], UAV_ALTITUDE)
    next_uav_position = (next_center[0], next_center[1], UAV_ALTITUDE)
    num_steps = 100 
    step_size = distance_to_next_center / num_steps
    for step in range(num_steps):
      interp_x = current_uav_position[0] + step_size * (step + 1) * (next_uav_position[0] - current_uav_position[0]) / distance_to_next_center
      interp_y = current_uav_position[1] + step_size * (step + 1) * (next_uav_position[1] - current_uav_position[1]) / distance_to_next_center
      uav_position = (interp_x, interp_y, UAV_ALTITUDE)
      current_uav_position = uav_position
      plt.plot([current_uav_position[0], next_uav_position[0]], [current_uav_position[1], next_uav_position[1]], color='black', linestyle='dotted')

    # 첫번째 클러스터 
    gu_positions = np.array([(gu['x'], gu['y']) for gu in current_gus])
    kmeans = KMeans(n_clusters=1, n_init="auto").fit(gu_positions)
    first_level_center = kmeans.cluster_centers_[0]
    first_level_gus = [{'x': pos[0], 'y': pos[1], 'cluster': 0, 'battery': gu_bat[i]} for i, pos in enumerate(gu_positions)]

    # GU가 완충 됐을 때 re-clusting
    while any(gu['battery'] < GU_MAXBATT for gu in first_level_gus):
    
        new_first_level_centers, new_first_level_gus = recluster_gus([first_level_center], first_level_gus)

        first_level_center = new_first_level_centers[0]
        first_level_gus = new_first_level_gus

plt.xlabel("x-axis [m]")
plt.ylabel("y-axis [m]")
plt.title("Simulation Environment")
plt.xticks(np.arange(X_MIN, X_MAX + 1, X_GRID))
plt.yticks(np.arange(Y_MIN, Y_MAX + 1, Y_GRID))
plt.xlim(X_MIN, X_MAX)
plt.ylim(Y_MIN, Y_MAX)
plt.grid()
plt.show()
