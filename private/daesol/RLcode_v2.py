import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.patches as mpatches
import openpyxl as xl
import pandas as pd
from scipy.spatial import distance_matrix
from sklearn.cluster import KMeans
from IPython.display import display, clear_output
import random
import time

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

def distance3d(center,i,j):
    return (((center[i][0]-center[j][0])**2)+((center[i][1]-center[j][1])**2)+100)**(1/2)

def qreward(sec,length,timestep,salerate):
    return (sec/length)*(salerate**timestep)

def qfunc(currents,nexts,length,qtable,alpha,gamma):
    if nexts>currents:
        currentq=qtable[currents][nexts-1][1]
    else:
        currentq=qtable[currents][nexts][1]
    nextq_array=[]
    for i in range(length):
        nextq_array.append(qtable[currents][i][1])
    min_nextq=np.min(nextq_array)
    return currentq+(alpha*((gamma*min_nextq)-(currentq)))

def qfunc2(currents,nexts,length,qtable,alpha,gamma,center,timestep,distance):
    distancei=(((center[int(currents)][0]-center[int(nexts)][0])**2)+((center[int(currents)][1]-center[int(nexts)][1])**2)+100)**(1/2)
    reward=distance*(gamma**timestep)
    if nexts>currents:
        currentq=qtable[currents][nexts-1][1]
    else:
        currentq=qtable[currents][nexts][1]
    nextq_array=[]
    for i in range(length):
        nextq_array.append(qtable[currents][i][1])
    min_nextq=np.min(nextq_array)
    return currentq+(alpha*(reward+(gamma*min_nextq)-(currentq)))


def total_used_time(totaldistance,speed,chargetime):
    return (totaldistance/speed)+chargetime

def calc_move_distances(route,center,length):
    total_distance=0
    for i in range(length):
        distancei=(((center[int(route[i])][0]-center[int(route[i+1])][0])**2)+((center[int(route[i])][1]-center[int(route[i+1])][1])**2)+100)**(1/2)
        total_distance+=distancei
        print(distancei)
    return total_distance


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

MAX_UAV_BATTERY=1200 #uav's battery; 500mW
UAV_SPEED = 6 #[m/s]
UAV_MOVE= 10 #10mWs when moving
UAV_HOV= 2 # 2mWs when hovering
UAV_LOWBATT = 350 # go to charge station when UAV's battery left is 200mW

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

#k-mean, finding cluster centers
countA=np.zeros(10)
clusterNum=0
RADIUS_FOR_KMEAN=MAX_BEAM_RADIUS #17m(radius) - x(constant)
for i in range(1,10):
    countA=np.zeros(10)
    kmeans = KMeans(
        n_clusters=i,
        n_init="auto"
    ).fit(gu_xyz)
    centers = kmeans.cluster_centers_
    clear_output(False)
    for j in range(i):
        for k in range(10):
            if centers[j][0]-RADIUS_FOR_KMEAN<=gu_x[k]<=centers[j][0]+RADIUS_FOR_KMEAN and centers[j][1]-RADIUS_FOR_KMEAN<=gu_y[k]<=centers[j][1]+RADIUS_FOR_KMEAN:
                    print("radius: "+str(RADIUS_FOR_KMEAN))
                    print("center: "+str(centers[j][0])+", "+str(centers[j][1]))
                    print("gu("+str(k)+"):"+str(gu_x[k])+", "+str(gu_y[k]))
                    countA[k]=1
    print("cluster num= "+str(i))
    print(centers)
    print(countA)
    if np.sum(countA)==10:
        clusterNum=i
        break
print(centers)
print(clusterNum)
print("count:"+str(countA))
print("----------------------------------")

centers=np.vstack(([[50, 50]], centers[:, 0:2]))


for i in range(len(centers)):
    plt.scatter(x=centers[i][0], y=centers[i][1], c=color[9])
    plt.text(x=centers[i][0] - 1.5, y=centers[i][1] + 2, s=f"C-{i}")


#RL
episode=500
#about route
route=[]
initial_state=list(range(len(centers)))
possible_state=initial_state
current_state=0

#about Q table - define func: 
# qvalue(sec,length,timestep,salerate), 
# sec= usedtime(distance,speed,chargetime)
EPSILON = 0.99 #epsilon decay
epsilon_decay=EPSILON
epsilon_decrease=EPSILON/episode
DISCOUNT_RATE=0.9
LEARNING_RATE=0.1
length_centers = len(centers)-1
NUM_POSSIBLE_STATE=(length_centers**2)+length_centers


qtable=np.zeros((len(centers),length_centers,3))
#Q table col-current, col-next 
route4input=list(range(len(centers)))
for inputcurrent in range(len(centers)):
    tmpnext=list(filter(lambda n: n!=inputcurrent,route4input))
    for i in range(len(centers)-1):
        qtable[inputcurrent][i][0]=tmpnext[i]
column_names = ["Current", "Next", "Q value", "Data #'s"]
m,n,r = qtable.shape
out_arr = np.column_stack((np.repeat(np.arange(m),n),qtable.reshape(m*n,-1)))
out_df = pd.DataFrame(out_arr,columns=column_names)
#qtable_df=pd.DataFrame(qtable,columns=column_names)

#episode_done=np.zeros((episode,len(centers)))
#episode_done_df=pd.DataFrame(episode_done)
episode_done=[] #2dimension

#get random route, max charge time
route=[]
currentbatt=MAX_UAV_BATTERY
gucounter=0
route.append(0)
possible_state.remove(0)
"""for a in range(9):"""
batt4p2p=0
batt4hov=0
batt4txpw=0
gucounter=0
chargingtime=0
gubigcounter=0
karray1=[]
#randomly pick next state using epsilon
for countepisode in range(episode):
    route=[]
    currentbatt=MAX_UAV_BATTERY
    gubigcounter=0
    possible_state=list(range(len(centers)))
    route.append(0)
    possible_state.remove(0)
    chargingtime=0
    karray1=[]
    timestep=1
    current_state=0
    while gubigcounter!=NUM_GU:
        batt4p2p=0
        batt4hov=0
        batt4txpw=0
        gucounter=0
        if currentbatt>UAV_LOWBATT: # if current battery is higher than
            if random.random()<epsilon_decay: #any states
                next_state=random.choice(possible_state)
                route.append(next_state)
                print("a")
            else: #state with highest Q value
                tmparray=[]
                for findhighest in range(length_centers):
                    x=(current_state*length_centers)+findhighest
                    tmparray.append(qtable[current_state][findhighest][2])
                next_state = int(qtable[current_state][np.argmin(tmparray)][0])    
                if np.in1d(possible_state,next_state).any()==0:
                    next_state=random.choice(possible_state)
                route.append(next_state)
                print(f"b, next={current_state}, tmparray={np.argmin(tmparray)}")
            batt4p2p=(distance3d(centers,current_state,next_state)/UAV_SPEED)*UAV_MOVE #power used for moving point to point
            print(f"current={current_state}, next={next_state}")
            d1=calc_move_distances(route,centers,len(route)-1)
            #find qvalue->qfunc() before changing state
            if next_state>current_state:
                qtable[current_state][next_state-1][1]=qfunc2(current_state,next_state,length_centers,qtable,LEARNING_RATE,DISCOUNT_RATE,centers,timestep,d1)
                timestep+=1  
            else:
                qtable[current_state][next_state][1]=qfunc2(current_state,next_state,length_centers,qtable,LEARNING_RATE,DISCOUNT_RATE,centers,timestep,d1)
                timestep+=1  
            current_state=next_state
            print(possible_state)
            if np.in1d(possible_state,current_state).any()==1:
                possible_state.remove(current_state)
            print(f"tmp route:{route}")
            print(f"batt4p2p={batt4p2p}")
    #memo
            currentbatt-=batt4p2p
            time1=[]
            tttt=1
            karray=[]
            #check how many GU's are there inside beam circle
            for k in range(10):
                if centers[current_state][0]-RADIUS_FOR_KMEAN<=gu_x[k]<=centers[current_state][0]+RADIUS_FOR_KMEAN and centers[current_state][1]-RADIUS_FOR_KMEAN<=gu_y[k]<=centers[current_state][1]+RADIUS_FOR_KMEAN and gu_bat[k]!=100:
                    if np.in1d(karray1,k).any()==True:
                        print(f"{k}is repeated")
                    else:
                        karray1.append(k)
                        karray.append(k)
                        gucounter+=1
                        gubigcounter+=1
                        dd=(((centers[current_state][0]-gu_x[k])**2)+((centers[current_state][1]-gu_y[k])**2)+100)**(1/2)
                        time1=np.append(time1,(100/calc_rx_power(dd)))
                    
            if len(time1)>0: #largest time
                tttt=int(np.max(time1))
                chargingtime+=tttt
                batt4hov=tttt*UAV_HOV # x second * power used for hovering 
                batt4txpw=gucounter*100 # number of GU's inside Beam * power used for trasmitting power
                print(f"batt4hov={batt4hov}, batt4txpw={batt4txpw}, time={tttt}, k={karray}")
                currentbatt-=(batt4hov+batt4txpw) 
                #ok until here
        else: #go to charge station
            batt4p2p=(distance3d(centers,current_state,0)/UAV_SPEED)*UAV_MOVE #power used for moving point to point
            current_state=0
            currentlocXY=centers[current_state]
            route.append(current_state)
            currentbatt=MAX_UAV_BATTERY
    #memo end
        print(f"currentbatt={int(currentbatt)}, routes={route}, gu={gucounter}")
  
    #find qvalue ->qreward
    total_d=calc_move_distances(route,centers,len(route)-1)
    total_t=total_used_time(total_d,UAV_SPEED,chargingtime)
    """for insertq in range(len(route)-1):
        cs=route[insertq]
        ns=route[insertq+1]
        if ns>cs:
            qtable[cs][ns-1][2]+=1
            qtable[cs][ns-1][1]=LEARNING_RATE*qreward(total_t,length_centers,insertq,DISCOUNT_RATE)
        else:
            qtable[cs][ns][2]+=1
            qtable[cs][ns][1]=LEARNING_RATE*qreward(total_t,length_centers,insertq,DISCOUNT_RATE)
"""
    print(f"total time={total_t}")
    print("--------------------------------")
    episode_done.append(route)
    epsilon_decay-=epsilon_decrease
#------------------------------------------------
#-----------------------------------------------
#---------------------------------------------
#--------------------------------------------
#for final route



m,n,r = qtable.shape
out_arr = np.column_stack((np.repeat(np.arange(m),n),qtable.reshape(m*n,-1)))
out_df = pd.DataFrame(out_arr,columns=column_names)

print("Q table:")
print(pd.DataFrame(out_df))
print(f"last route: {episode_done[episode-1]}")
print(f"centers: {centers}")
new_row = pd.DataFrame([['episode:'+str(episode), '-', '-','-']], columns = out_df.columns)
new_df = pd.concat([out_df.iloc[:0], new_row, out_df.iloc[0:]], ignore_index = True)
new_df.to_csv('qtable.csv')


columns = ['x_coordinate', 'y_coordinate']
df = pd.DataFrame(centers, columns=columns)
df.to_csv('centers.csv')

columns = ['route']
df = pd.DataFrame(episode_done[episode-1], columns=columns)
df.to_csv('routes.csv')


route = episode_done[episode-1]

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
