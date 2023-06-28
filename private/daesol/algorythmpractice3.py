import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.patches as mpatches
from scipy.spatial import distance_matrix

def makeUAV(xpos,ypos,maxbeam,uavno):
    uav_x_pos = xpos  # x position of uav 0,1,...,GRID_SIZE-1
    uav_y_pos = ypos  # y position of uav 0,1,...,GRID_SIZE-1
    uav_number = "UAV"+str(uavno)
    uav_x = GRID[1, uav_x_pos, uav_y_pos]
    uav_y = GRID[0, uav_x_pos, uav_y_pos]
    uav_z2 = UAV_ALTITUDE  # uav's altitude [meter]

    plt.scatter(x=uav_x, y=uav_y, c="green")
    plt.text(x=uav_x - 3.5, y=uav_y - 4, s=uav_number)

def makeBeamCircle(xpos,ypos,maxbeam,color):
    uav_x_pos = xpos  # x position of uav 0,1,...,GRID_SIZE-1
    uav_y_pos = ypos  # y position of uav 0,1,...,GRID_SIZE-1
    uav_x = GRID[1, uav_x_pos, uav_y_pos]
    uav_y = GRID[0, uav_x_pos, uav_y_pos]
    uav_z2 = UAV_ALTITUDE  # uav's altitude [meter]

    beam_circle = Ellipse(
        xy=(uav_x, uav_y),
        width=maxbeam,
        height=maxbeam,
        angle=0,
        edgecolor="none",
        facecolor=color,
        alpha=0.2,
    )
    uav_beam = ax.add_patch(beam_circle)

def searchGu(x,y,uavno):
    uavx=(x+0.5)*10
    uavy=(y+0.5)*10
    tmp_count=0
    count=0
    for k in range(10):
        if uavx<=gu_x[k]<=uavx+17 and uavy<=gu_y[k]<=uavy+17: #quadrant1
            print("quadrant1: ",end='')
            print(k)
            gu_memory[uavno][0][k]=gu_x[k]
            gu_memory[uavno][1][k]=gu_y[k]
        if uavx-17<=gu_x[k]<=uavx and uavy<=gu_y[k]<=uavy+17: #quadrant2
            print("quadrant2: ",end='')
            print(k)
            gu_memory[uavno][0][k]=gu_x[k]
            gu_memory[uavno][1][k]=gu_y[k]
        if uavx-17<=gu_x[k]<=uavx and uavy-17<=gu_y[k]<=uavy: #quadrant3
            print("quadrant3: ",end='')
            print(k)
            gu_memory[uavno][0][k]=gu_x[k]
            gu_memory[uavno][1][k]=gu_y[k]
        if uavx<=gu_x[k]<=uavx+17 and uavy-17<=gu_y[k]<=uavy: #quadrant4
            print("quadrant4: ",end='')
            print(k)
            gu_memory[uavno][0][k]=gu_x[k]
            gu_memory[uavno][1][k]=gu_y[k]
    #print(gu_memory)
    for k in range(10):
        if gu_memory[uavno][0][k]!=11:
            if count==0:
                plt.plot([uavx,gu_memory[uavno][0][k]],[uavy,gu_memory[uavno][1][k]],color="red")
                tmp_state[tmp_count][0]=gu_memory[uavno][0][k]
                tmp_state[tmp_count][1]=gu_memory[uavno][1][k]
                #print(tmp_state[tmp_count])
                count+=1
            else:
                #print(tmp_state[tmp_count])
                plt.plot([tmp_state[tmp_count][0],gu_memory[uavno][0][k]],[tmp_state[tmp_count][1],gu_memory[uavno][1][k]],color="red")
                tmp_count+=1
                tmp_state[tmp_count][0]=gu_memory[uavno][0][k]
                tmp_state[tmp_count][1]=gu_memory[uavno][1][k]
                count+=1
        


#intial parameter
NUM_GU = 10  # number of ground users

X_MIN = 0  # minimum x-axis [meter]
X_MAX = 100  # maximum x-axis [meter]
Y_MIN = 0  # minimum y-axis [meter]
Y_MAX = 100  # maximum y-axis [mseter]

UAV_ALTITUDE = 10  # altitude of uav [meter]
MAX_BEAM_ANGLE = 60  # maximum beamforming angle [degree]

# maximum beamforming diameter [meter]
MAX_BEAM_DIAMETER = 2*UAV_ALTITUDE*np.tan(MAX_BEAM_ANGLE*np.pi/180)


X_GRID = 10  # number of x grid
Y_GRID = 10  # number of y grid


UAV_TX_POWER = 30  # uav's transmit power in [dBm]

#initial variables
t = 0  # time [seconds]

gu_bat = np.zeros((NUM_GU,)) # battery of ground user [mWh]

#generate variables
# generate ground user location randomly x,y,z [meters]
gu_x = np.random.uniform(low=X_MIN, high=X_MAX, size=(NUM_GU,))
gu_y = np.random.uniform(low=Y_MIN, high=Y_MAX, size=(NUM_GU,))
gu_z = np.zeros((NUM_GU,))

# print
gu_x, gu_y, gu_z

# generate meshgrid
tmp_x = np.linspace(X_MIN + X_GRID/2,X_MAX-X_GRID/2,X_GRID)
tmp_y = np.linspace(Y_MIN + Y_GRID/2,Y_MAX-Y_GRID/2,Y_GRID)
GRID = np.array(np.meshgrid(tmp_x, tmp_y))

GRID

# generate uav location

uav_x_pos = 1  # x position of uav 0,1,...,GRID_SIZE-1
uav_y_pos = 0  # y position of uav 0,1,...,GRID_SIZE-1

uav_x = GRID[1, uav_x_pos, uav_y_pos]
uav_y = GRID[0, uav_x_pos, uav_y_pos]
uav_z = UAV_ALTITUDE  # uav's altitude [meter]

# print
uav_x, uav_y, uav_z
fig, ax = plt.subplots(figsize=(6, 6))




for i in range(NUM_GU):
    plt.scatter(x=gu_x[i], y=gu_y[i], c="blue")
    plt.text(x=gu_x[i] - 3.5, y=gu_y[i] - 4, s=f"GU-{i}")
    plt.text(x=gu_x[i] - 6, y=gu_y[i] - 7, s=f"{gu_bat[i]}mWh")

#if uav altitude=10m, max beam diameter=34.64m, radius=17m
#whole grid=100x100 m, 1 grid = 10 x 10 m
#gu's # = 10
#uav x,y pos => 0~9, aligned center(+0.5)
#ex. {0,1} -> {0.5, 1.5} -> {5 m, 15 m}
print(MAX_BEAM_DIAMETER)
print(gu_x)
print(gu_y)
print(gu_z)
gu_memory=np.ones((25,2,10)) + 10
tmp_state=np.ones((10,2)) 

countUAV=1
for i in range(1,11,2):
    if i==1 or i==5 or i==9:
        for k in range(0,10,2):
            makeUAV(i,k,MAX_BEAM_DIAMETER,countUAV)
            if countUAV%2 == 0:
                makeBeamCircle(i,k,MAX_BEAM_DIAMETER,"yellow")
            else:
                makeBeamCircle(i,k,MAX_BEAM_DIAMETER,"orange")
            if countUAV%5 !=0:
                plt.plot([(i+0.5)*10,(i+0.5)*10],[(k+0.5)*10,(k+2.5)*10],color="green")
            elif countUAV==25:
                searchGu(i,k,countUAV-1)
            else:
                plt.plot([(i+0.5)*10,(i+2.5)*10],[(k+0.5)*10,(k+0.5)*10],color="green")
            print(countUAV)
            searchGu(i,k,countUAV-1)
            countUAV+=1
            
            """gu_memory[0]=11
            gu_memory[1]=11
            for a in range(10):
                tmp_state[a]=1"""
            
    else:
        for k in range(8,-2,-2):
            makeUAV(i,k,MAX_BEAM_DIAMETER,countUAV)
            if countUAV%2 == 0:
                makeBeamCircle(i,k,MAX_BEAM_DIAMETER,"yellow")
            else:
                makeBeamCircle(i,k,MAX_BEAM_DIAMETER,"orange")
            if countUAV%5 ==0:    
                plt.plot([(i+0.5)*10,(i+2.5)*10],[(k+0.5)*10,(k+0.5)*10],color="green")
            else:
                plt.plot([(i+0.5)*10,(i+0.5)*10],[(k+0.5)*10,(k-1.5)*10],color="green")
            print(countUAV)
            searchGu(i,k,countUAV-1)
            countUAV+=1
            
            """gu_memory[0]=11
            gu_memory[1]=11
            for a in range(10):
                tmp_state[a]=1"""

print(gu_memory)
plt.xlabel("x-axis [m]")
plt.ylabel("y-axis [m]")
plt.title("Simulation Environment")
plt.xticks(np.arange(X_MIN, X_MAX + 1, X_GRID))
plt.yticks(np.arange(Y_MIN, Y_MAX + 1, Y_GRID))
plt.xlim(X_MIN, X_MAX)
plt.ylim(Y_MIN, Y_MAX)
plt.grid()
plt.show()
