import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.patches as mpatches
import openpyxl as xl
import pandas as pd
from scipy.spatial import distance_matrix
from sklearn.cluster import KMeans
from IPython.display import display, clear_output

def makeUAV(xpos,ypos,maxbeam,uavno):
    uav_number = "UAV"+str(uavno)
    plt.scatter(x=xpos, y=ypos, c="green")
    plt.text(x=xpos - 3.5, y=ypos - 4, s=uav_number)

def makeBeamCircle(xpos,ypos,maxbeam,color):
    beam_circle = Ellipse(
        xy=(xpos, ypos),
        width=maxbeam,
        height=maxbeam,
        angle=0,
        edgecolor="none",
        facecolor=color,
        alpha=0.2,
    )
    uav_beam = ax.add_patch(beam_circle)

def makeBeamCirclewDot(xpos,ypos,maxbeam,color):
    beam_circle = Ellipse(
        xy=(xpos, ypos),
        width=maxbeam,
        height=maxbeam,
        angle=0,
        edgecolor="none",
        facecolor=color,
        alpha=0.2,
    )
    uav_beam = ax.add_patch(beam_circle)
    plt.scatter(xpos,ypos, c="yellow")

def calc_rx_power(d):
    # received power [mWh]
    return 1*10**((TX_POWER - (20*np.log10((4*np.pi*d*F)/C)))/10) * 1000

def distance3d(center,i,j):
    return (((center[i][0]-center[j][0])**2)+((center[i][1]-center[j][1])**2)+100)**(1/2)
def distance2(center,gux,guy,i,j):
    return (((center[i][0]-gux[j][0])**2)+((center[i][1]-guy[j][1])**2)+100)**(1/2)

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
MAX_BEAM_DISTANCE = UAV_ALTITUDE / np.cos(MAX_BEAM_ANGLE * np.pi / 180)
MAX_BEAM_DIAMETER = 2*UAV_ALTITUDE*np.tan(MAX_BEAM_ANGLE*np.pi/180)
MAX_BEAM_RADIUS = MAX_BEAM_DIAMETER/2
X_GRID = 10  # number of x grid
Y_GRID = 10  # number of y grid
UAV_TX_POWER = 30  # uav's transmit power in [dBm]

MAX_UAV_BATTERY=1200 #uav's battery; 500mW
UAV_SPEED = 6 #[m/s]
UAV_MOVE= 10 #10mWs when moving
UAV_HOV= 2 # 5mWs when hovering
UAV_LOWBATT = 250 # go to charge station when UAV's battery left is 200mW

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

gu_memory=np.ones((2,NUM_GU))
df=pd.read_excel('locationInformation.xlsx')
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

countA=np.zeros(10)
clusterNum=0
RADIUS_FOR_KMEAN=MAX_BEAM_RADIUS #17m(radius) - x(constant)

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
route=[0, 6, 1, 0, 2, 4, 5,0]
points=np.vstack((centers))
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
uavbat1=MAX_UAV_BATTERY
t2=0
ttt=0
# plot real time update trajectory
for t in range(len(uav_time)-1):
    ttt=t+t2
    # update title
    ax.set_title(f"Simulation Result [ t = {ttt}s ]")

    # calculate distance
    distance_uav2gu = distance_matrix(
        [np.append(uav_time[t+1], UAV_ALTITUDE)], gu_xyz)

    # calculate receive power
    rx_power = calc_rx_power(distance_uav2gu)

    # update the gu battery base on the maximum distance
    for a in range(1,6):
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
            t2+=int(np.max(times))
            
            ax.text(x=uav_time[t+1][0] - 6, y=uav_time[t+1][1] - 8, s=f"{int(np.max(times)):.2f}sec")
            
            uavbat1-=(gucount*100)+(np.max(times))
            #gu_bat += (rx_power*(distance_uav2gu <= MAX_BEAM_DISTANCE+1))[0]
    if uav_time[t+1][0]==50 and uav_time[t+1][1]==50:
        uavbat1=MAX_UAV_BATTERY
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

    
    if t>0:
        uav_batt.remove()
    uav_batt = ax.text(
        x=uav_time[t+1][0] - 6, y=uav_time[t+1][1] - 7, s=f"{uavbat1:.2f}mWs")

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




