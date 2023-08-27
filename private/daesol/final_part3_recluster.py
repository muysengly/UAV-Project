import time
import math
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, clear_output
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
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

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

def calc_minhovtime(i):
    distance_uav2gu = np.squeeze(distance_matrix(
                [np.append(centers[i], UAV_ALTITUDE)], gu_xyz))
    
    return 100/calc_rx_power(distance_uav2gu[np.argmin(distance_uav2gu)])

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
UAV_MOVE= 10 #10mWs when moving
UAV_HOV= 5 # 5mWs when hovering
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
####################################################################
#call GU
gu_memory=np.ones((2,NUM_GU))
gu_x = np.random.uniform(low=X_MIN, high=X_MAX, size=(NUM_GU,))
gu_y = np.random.uniform(low=Y_MIN, high=Y_MAX, size=(NUM_GU,))
df=pd.read_excel('locationInformation.xlsx')
for x in range(2):
    for y in range(10):
        gu_memory[x][y]=int(df.iloc[y+1,x])
gu_x = gu_memory[0]
gu_y = gu_memory[1]
gu_z = np.zeros((NUM_GU,))
#gu_z = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
gu_xyz = np.array((gu_x,gu_y,gu_z)).T

current_batt = []
for i in range(NUM_GU):
    ax.scatter(x=gu_x[i], y=gu_y[i], c="blue")
    ax.text(x=gu_x[i] - 3.5, y=gu_y[i] - 4, s=f"GU-{i}")
    current_batt.append(
        ax.text(x=gu_x[i] - 6, y=gu_y[i] - 7, s=f"{gu_bat[i]}mWh"))
####################################################################
#K-mean clustering

######find minimum number of cluster
countA=np.zeros(10)
minClusterNum=0
RADIUS_FOR_KMEAN=MAX_BEAM_RADIUS #17m(radius) - x(constant)
for i in range(1,10):
    countA=np.zeros(10)
    kmeans = KMeans(n_clusters=i, n_init="auto").fit(gu_xyz)
    centers = kmeans.cluster_centers_
    clear_output(False)
    for j in range(i):
        for k in range(10):
            if centers[j][0]-RADIUS_FOR_KMEAN<=gu_x[k]<=centers[j][0]+RADIUS_FOR_KMEAN and centers[j][1]-RADIUS_FOR_KMEAN<=gu_y[k]<=centers[j][1]+RADIUS_FOR_KMEAN:
                    countA[k]=1
    print("cluster num= "+str(i))
    print(centers)
    print(countA)
    if np.sum(countA)==10:
        minClusterNum=i
        break
print(centers)
print(f"minimum cluster #:{minClusterNum}")

#####################################################################
###########Mode 2 -> re-clustering
num=[]
dist=[]
T_taken=[]
T_total=[]
RU=[]
C_all=[]

#for n in range(NUM_GU,(minClusterNum-1),-1):
for n in range(NUM_GU,(minClusterNum-1),-1):
    total_time=0
    kmeans = KMeans(n_clusters=n, n_init="auto").fit(gu_xyz)
    centers = kmeans.cluster_centers_
    clear_output()
    centers=np.vstack(([[50, 50]], centers[:, 0:2]))
    print(f"before:")
    print("centers")
    print(centers)

    countRoute=1
    countCenter=len(centers)-1
    shortest_distance, route = tsp_dp(centers)

    gu_charged=[0,0,0,0,0,0,0,0,0,0]
    gu_full=[1,1,1,1,1,1,1,1,1,1]
    identifyGU=0
    print(f"route first:{route}")
    while True:
        distance_uav2gu = np.squeeze(distance_matrix(
                    [np.append(centers[route[identifyGU]], UAV_ALTITUDE)], gu_xyz))
        distance_center=np.squeeze(distance_uav2gu <= MAX_BEAM_DISTANCE)
        distance_index=np.where(distance_center == True)
        print(len(distance_index[0]))
        print(f"distance matrix {distance_index[0]}")
        print(f"loc now:{route[identifyGU]}")
        total_time+=(100/calc_rx_power(distance_uav2gu[np.argmin(distance_uav2gu)]))
        if len(distance_index[0])>2:
            new_gu=gu_xyz[distance_index[0]]
            insert_route=[]
            gu_charged[distance_index[0][0]]=1
            print(f"new gu:{new_gu}")
            new_gu=np.delete(new_gu,0,axis=0)
            print(f"new gu after={new_gu}")
            kmeans = KMeans(n_clusters=1, n_init="auto").fit(new_gu)
            centers2 = kmeans.cluster_centers_
            print(f"centers2{centers2}")
            centers=np.vstack((centers, centers2[:, 0:2]))
            #add centers
            countCenter+=1
            #add routes
            route=np.insert(route,countRoute,countCenter)
            countRoute+=1
            print(f"centers={centers}")
            print(f"routes: {route}")
            total_time+=(100/calc_rx_power(distance_uav2gu[np.argmin(distance_uav2gu)]))
            identifyGU+=1
        else:
            for x in range(len(distance_index[0])):
                gu_charged[distance_index[0][x]]=1
        if route[identifyGU]==0 and countRoute>3:
            break
        countRoute+=1
        identifyGU+=1
    shortest_distance, route = tsp_dp(centers)
    total_time+=shortest_distance/UAV_SPEED
    num.append(n)
    dist.append(shortest_distance)
    T_total.append(total_time)
    RU.append(route)
    C_all.append(centers)


###############################################################
#results
dictionary={
    'Cluster_num' : num,
    'DIST' : dist,
    'Total' : T_total,
    'ROUTE' : RU
}

df=pd.DataFrame(dictionary)
df.to_csv('result_part3.csv', index=False)

print(df)
print("df min")
print(np.argmin(df['Total']))
print(df.loc[np.argmin(df['Total'])])
print("centers")
print(C_all[np.argmin(df['Total'])])

route=RU[np.argmin(df['Total'])]
centers=C_all[np.argmin(df['Total'])]

points=centers

print("after")
print(centers)
print(route)

#######################################################################


for i in range(len(centers)):
    plt.scatter(x=centers[i][0], y=centers[i][1], c=color[9])
    plt.text(x=centers[i][0] - 1.5, y=centers[i][1] + 2, s=f"C-{i}")

#uav fly


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
for i in range(NUM_GU):
    gu_bat[i]=0
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

# design the axis
ax.set_xlabel("x-axis [m]")
ax.set_ylabel("y-axis [m]")
ax.set_xticks(np.arange(X_MIN, X_MAX + 1, X_GRID))
ax.set_yticks(np.arange(Y_MIN, Y_MAX + 1, Y_GRID))
ax.set_xlim(X_MIN, X_MAX)
ax.set_ylim(Y_MIN, Y_MAX)
ax.grid()
uavbat1=1200
t2=0
ttt=0
# plot real time update trajectory
for t in range(len(uav_time)-1):
    ttt=t+t2
    # update title
    ax.set_title(f"Simulation Result")

    # calculate distance
    distance_uav2gu = distance_matrix(
        [np.append(uav_time[t+1], UAV_ALTITUDE)], gu_xyz)

    # calculate receive power
    rx_power = calc_rx_power(distance_uav2gu)

    # update the gu battery base on the maximum distance
    for a in range(1,len(centers)):
        gucount=0
        charge=0
        times=[]
        if centers[a][0]==uav_time[t+1][0] and centers[a][1]==uav_time[t+1][1]:
            for k in range(10):
                if centers[a][0]-RADIUS_FOR_KMEAN<=gu_x[k]<=centers[a][0]+RADIUS_FOR_KMEAN and centers[a][1]-RADIUS_FOR_KMEAN<=gu_y[k]<=centers[a][1]+RADIUS_FOR_KMEAN and gu_bat[k]!=100:
                    gu_bat[k]=100
                    gucount+=1
                    d1=(((centers[a][0]-gu_x[k])**2)+((centers[a][1]-gu_y[k])**2)+100)**(1/2)
                    times=np.append(times,(100/calc_rx_power(d1)))
                    #print("centers:"+str(centers[a])+", gus:"+str(gu_x[k])+","+str(gu_y[k]))
                    #print(times)
            
            #gu_bat += (rx_power*(distance_uav2gu <= MAX_BEAM_DISTANCE+1))[0]
    if uav_time[t+1][0]==50 and uav_time[t+1][1]==50:
        uavbat1=1200
            #gu_bat+=(100*(distance_uav2gu<=MAX_BEAM_DISTANCE+1))[0]
        
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
    uavbat1-=UAV_MOVE

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






"""
#make cluster center points
for i in range(len(centers)):
    plt.scatter(x=centers[i][0], y=centers[i][1], c=color[9])
    plt.text(x=centers[i][0] - 1.5, y=centers[i][1] + 2, s=f"C-{i}")


colorcount=0
for i in range(len(route)-1):
    x_vals = [centers[int(route[i])][0], centers[int(route[i + 1])][0]]
    y_vals = [centers[int(route[i])][1], centers[int(route[i + 1])][1]]
    if colorcount==8:
        colorcount=0
    plt.plot(x_vals, y_vals, color=color[colorcount])
    colorcount+=1

plt.xlabel("x-axis [m]")
plt.ylabel("y-axis [m]")
plt.title("Simulation Environment")
plt.xticks(np.arange(X_MIN, X_MAX + 1, X_GRID))
plt.yticks(np.arange(Y_MIN, Y_MAX + 1, Y_GRID))
plt.xlim(X_MIN, X_MAX)
plt.ylim(Y_MIN, Y_MAX)
plt.grid()
plt.show()
"""