# UAV velocity = 1m/s
#uavx=(x+0.5)*10,uavy=(y+0.5)*10
 #if uavx-17<=gu_x[k]<=uavx+17 and uavy-17<=gu_y[k]<=uavy+17:

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.patches as mpatches
import openpyxl as xl
import pandas as pd
from scipy.spatial import distance_matrix
from sklearn.cluster import KMeans
from IPython.display import display, clear_output

def has_duplicates(seq):
    return len(seq) != len(set(seq))

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

#initial variables
t = 0  # time [seconds]

gu_bat = np.zeros((NUM_GU,)) # battery of ground user [mWh]

#call information from excel
gu_memory=np.ones((2,10))
df=pd.read_excel('locationInformation.xlsx')
for x in range(2):
    for y in range(10):
        gu_memory[x][y]=int(df.iloc[y+1,x])
print(gu_memory)

# generate meshgrid
tmp_x = np.linspace(X_MIN + X_GRID/2,X_MAX-X_GRID/2,X_GRID)
tmp_y = np.linspace(Y_MIN + Y_GRID/2,Y_MAX-Y_GRID/2,Y_GRID)
GRID = np.array(np.meshgrid(tmp_x, tmp_y))

GRID
fig, ax = plt.subplots(figsize=(6, 6))

guUAVdistance=np.zeros(10)
current_batt = []
for i in range(NUM_GU):
    ax.scatter(x=gu_memory[0][i], y=gu_memory[1][i], c="blue")
    ax.text(x=gu_memory[0][i] - 3.5, y=gu_memory[1][i] - 4, s=f"GU-{i}")
    current_batt.append(
        ax.text(x=gu_memory[0][i] - 6, y=gu_memory[1][i] - 7, s=f"{gu_bat[i]:.2f}mWs"))
"""for i in range(NUM_GU):
    plt.scatter(x=gu_memory[0][i], y=gu_memory[1][i], c="blue")
    plt.text(x=gu_memory[0][i] - 3.5, y=gu_memory[1][i] - 4, s=f"GU-{i}")
    plt.text(x=gu_memory[0][i] - 6, y=gu_memory[1][i] - 7, s=f"{gu_bat[i]}mWh")"""

gu_z = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
gu_xyz = np.array((gu_memory[0],gu_memory[1],gu_z)).T
countA=np.zeros(10)
clusterNum=0
for i in range(1,9):
    kmeans = KMeans(
        n_clusters=i,
        n_init="auto"
    ).fit(gu_xyz)
    centers = kmeans.cluster_centers_
    clear_output(False)
    for j in range(i):
        for k in range(10):
            if centers[j][0]-MAX_BEAM_RADIUS<=gu_memory[0][k]<=centers[j][0]+MAX_BEAM_RADIUS and centers[j][1]-MAX_BEAM_RADIUS<=gu_memory[1][k]<=centers[j][1]+MAX_BEAM_RADIUS:
                countA[k]=1
    print("cluster num= "+str(i))
    print(centers)
    print(countA)
    if np.sum(countA)==10:
        clusterNum=i
        break

print(centers)
print(clusterNum)

#finding groups end

#code related to charge start
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

#first step
distancesFromNow = np.zeros(clusterNum)
currentXYloc=np.zeros(2)+50
for i in range(clusterNum):
    distancesFromNow[i]=(((centers[i][0] - currentXYloc[0])**2)+((centers[i][1] - currentXYloc[1])**2))**(1/2) 


print(distancesFromNow)
print(np.argmin(distancesFromNow))
nextloc=np.argmax(distancesFromNow)
#distancesFromNow[nextloc]=100
#nextloc=np.argmin(distancesFromNow)

currentXYloc=centers[nextloc]

print(currentXYloc)
time1=np.zeros(10)

for k in range(10):
    if currentXYloc[0]-MAX_BEAM_RADIUS<=gu_memory[0][k]<=currentXYloc[0]+MAX_BEAM_RADIUS and currentXYloc[1]-MAX_BEAM_RADIUS<=gu_memory[1][k]<=currentXYloc[1]+MAX_BEAM_RADIUS:
        print(k)
        distance1=(((gu_memory[0][k] - currentXYloc[0])**2)+((gu_memory[1][k] - currentXYloc[1])**2)+100)**(1/2)
        print("distance: "+str(distance1))
        while gu_bat[k]<100:
            gu_bat[k]+=calc_rx_power(distance1)
            time1[k]+=1
            #print("gumemory="+str(gu_memory[0][k]+", "+str(gu_memory))
            print(str(time1)+"second")
            #print(str(distance1)+"m, rx_power="+str(calc_rx_power(distance1)))
            print(str(gu_bat[k])+"mWs")
            n=input('continue?')
            if n=='a':
                #plt.text(x=centers[i][0] + 3.5, y=centers[i][1] + 4, s=f"UAV state-{i}")
            else:
                quit()
        current_batt[k].remove()
        current_batt[k] = ax.text(x=gu_memory[0][k] - 6, y=gu_memory[1][k] - 7, s=f"{gu_bat[k]:.2f}mWs")
        print(time1[k])
print(time1)
#first step end 



plt.xlabel("x-axis [m]")
plt.ylabel("y-axis [m]")
plt.title("Simulation Environment")
plt.xticks(np.arange(X_MIN, X_MAX + 1, X_GRID))
plt.yticks(np.arange(Y_MIN, Y_MAX + 1, Y_GRID))
plt.xlim(X_MIN, X_MAX)
plt.ylim(Y_MIN, Y_MAX)
plt.grid()
plt.show()

