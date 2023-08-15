import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.patches as mpatches
import openpyxl as xl
import pandas as pd
from scipy.spatial import distance_matrix


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
X_GRID = 10  # number of x grid
Y_GRID = 10  # number of y grid
UAV_TX_POWER = 30  # uav's transmit power in [dBm]

MAX_UAV_BATTERY=800 #uav's battery; 500mW
UAV_SPEED = 6 #[m/s]
UAV_MOVE= 10 #10mWs when moving
UAV_HOV= 5 # 5mWs when hovering
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
#input: centers: cluster centers array(x,y)
#output: route array

route=[0,3,5,2,1]

#drawing
for i in range(len(route)-1):
    x_vals = [centers[int(route[i])][0], centers[int(route[i + 1])][0]]
    y_vals = [centers[int(route[i])][1], centers[int(route[i + 1])][1]]
    plt.plot(x_vals, y_vals, color=color[i])




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
countA=np.zeros(10)
clusterNum=0
RADIUS_FOR_KMEAN=MAX_BEAM_RADIUS-10 #17m(radius) - x(constant)
for i in range(1,9):
    kmeans = KMeans(
        n_clusters=i,
        n_init="auto"
    ).fit(gu_xyz)
    centers = kmeans.cluster_centers_
    clear_output(False)
    for j in range(i):
        for k in range(10):
            if centers[j][0]-RADIUS_FOR_KMEAN<=gu_memory[0][k]<=centers[j][0]+RADIUS_FOR_KMEAN and centers[j][1]-RADIUS_FOR_KMEAN<=gu_memory[1][k]<=centers[j][1]+RADIUS_FOR_KMEAN:
                countA[k]=1
    print("cluster num= "+str(i))
    print(centers)
    print(countA)
    if np.sum(countA)==10:
        clusterNum=i
        break
print(centers)
print(clusterNum)"""

#distanceAB=(((gu_memory[0][nextloc]-gu_memory[0][currentloc])**2)+((gu_memory[1][nextloc]-gu_memory[1][currentloc])**2))**(1/2) 

"""for i in range(clusterNum):
    n=input('continue?')
    if n=='a':
        print(i)
        makeBeamCirclewDot(centers[i][0],centers[i][1],MAX_BEAM_DIAMETER,'orange')
        plt.text(x=centers[i][0] + 3.5, y=centers[i][1] + 4, s=f"UAV state-{i}")
    else:
        quit()
"""
"""clusterNum=7
kmeans = KMeans(
    n_clusters=clusterNum,
    n_init="auto"
).fit(gu_xyz)
centers = kmeans.cluster_centers_
clear_output(False)"""