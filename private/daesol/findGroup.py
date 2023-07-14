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

for i in range(NUM_GU):
    plt.scatter(x=gu_memory[0][i], y=gu_memory[1][i], c="blue")
    plt.text(x=gu_memory[0][i] - 3.5, y=gu_memory[1][i] - 4, s=f"GU-{i}")
    plt.text(x=gu_memory[0][i] - 6, y=gu_memory[1][i] - 7, s=f"{gu_bat[i]}mWh")


#makeUAV(95,85,MAX_BEAM_DIAMETER,1)
currentloc=int(df.iloc[0,2])

sxpos = 100
sypos = 100
trial = int(X_MAX/10)+1
groupArea=np.zeros((10,10))+11
bigcounter=0
counter1=0
counter2=0
testset=[6,5,4,11,11,11,11]
uavspot=np.zeros((10,2))
for x in range(trial):
    sypos=100
    for y in range(trial):
        counter1=0
        tmp_groupArea=np.zeros(10)+11
        for i in range(10):
            if sxpos-MAX_BEAM_RADIUS<=gu_memory[0][i]<=sxpos+MAX_BEAM_RADIUS and sypos-MAX_BEAM_RADIUS<=gu_memory[1][i]<=sypos+MAX_BEAM_RADIUS:
                #print(str(i)+", ",end='')
                tmp_groupArea[counter1]=int(i)
                counter1+=1
        print()
        
        print(str(sxpos)+", "+str(sypos))
        for a in range(10):
            intersection=len(set(groupArea[a])^set(tmp_groupArea))
            lengroup=len(set(groupArea[a]))
            lentmp=len(set(tmp_groupArea))
            print("inter: "+str(intersection))
            print("repeat: "+str(a))
            #print(np.unique(tmp_groupArea))
            if np.array_equal(groupArea[a],tmp_groupArea)==1:
                    print("case1: the same array. Pass")
                    break
            elif len(np.unique(tmp_groupArea)) == 1:
                print("case2: tmp has only 11's. Pass")
                break
            elif np.array_equal(groupArea[a],tmp_groupArea)==0:
                print("case3: two are different")
                print(len(np.intersect1d(groupArea[a],tmp_groupArea)))
                if len(np.intersect1d(groupArea[a],tmp_groupArea))>1:
                    print("case 3-1: there are more than one intersection")
                    if lengroup<lentmp:
                        print("case 3-1-1: update")
                        groupArea[a]=tmp_groupArea
                        uavspot[a][0]=sxpos
                        uavspot[a][1]=sypos
                        print(uavspot[a])
                        break
                    elif lengroup>lentmp:
                        print("case 3-1-2: stay")
                        break
                elif len(np.intersect1d(groupArea[a],tmp_groupArea))==1:
                    x=np.unique(groupArea[a])
                    y=np.unique(tmp_groupArea)
                    print("case 3-2: no intersection")
                    if len(np.unique(np.union1d(x,y)))-1==1:
                        print("case 3-2-1: only one different. Update in empty array")
                        groupArea[a]=tmp_groupArea
                        uavspot[a][0]=sxpos
                        uavspot[a][1]=sypos
                        print(uavspot[a])
                        break
                    elif len(np.unique(np.union1d(x,y)))-1>1:
                        print("case 3-2-2: two diff. Go to next array")
                        continue
        print(tmp_groupArea)
        print(groupArea)
        #print(uavspot)
        sypos-=10
        """n=input('continue? '+str(sxpos)+', '+str(sypos+10))
        if n=='a':
            continue
        else:
            quit()"""
        
    sxpos-=10
"""for a in range(10):
     for b in range(10):
          if a==b:
               continue
          else:
            if len(np.intersect1d(groupArea[a],groupArea[b]))>1:
                                groupArea[b]=11
                                uavspot[b][0]=0
                                uavspot[b][1]=0"""
print(groupArea)
print(uavspot)
countareas=10
#for count in range(10):
#     if np.array_equal(groupArea[count],[11,11,11,11,11,11,11,11,11,11])==0:
#          countareas+=1
print(countareas)
for count2 in range(countareas):
    makeBeamCirclewDot(uavspot[count2][0],uavspot[count2][1],MAX_BEAM_DIAMETER,'skyblue')
print(MAX_BEAM_RADIUS)
#find distance of gu-UAV
count12=1
nextloc=0
guUAVdistanceSum=0
stackcount=0
 



plt.xlabel("x-axis [m]")
plt.ylabel("y-axis [m]")
plt.title("Simulation Environment")
plt.xticks(np.arange(X_MIN, X_MAX + 1, X_GRID))
plt.yticks(np.arange(Y_MIN, Y_MAX + 1, Y_GRID))
plt.xlim(X_MIN, X_MAX)
plt.ylim(Y_MIN, Y_MAX)
plt.grid()
plt.show()

"""for x in range(9):
    stepCount+=1
    #plt.text(x=sxpos,y=sypos,s=str(stepCount))
    sxpos-=10
    for i in range(10):
        if sxpos-17<=gu_memory[0][i]<=sxpos+17 and sypos-17<=gu_memory[1][i]<=sypos+17:
            if foundUAVno[i]!=1:
                foundCount+=1
            foundUAVno[i]=1"""
"""
for b in range(8):
    for c in range(7):
        count12=1
        currentloc=int(df.iloc[0,2])
        nextloc=0
        guUAVdistanceSum=0
        for x in range(2):
            for y in range(10):
                gu_memory[x][y]=int(df.iloc[y+1,x])
        for a in range(9):
            #distanceAB=(((gu_memory[0][nextloc]-gu_memory[0][currentloc])**2)+((gu_memory[1][nextloc]-gu_memory[1][currentloc])**2))**(1/2) 
            #find distance of gu-uav #2
            print("#"+str(count12))
            count12+=1
            print("gumemory")
            print(gu_memory)

            for x in range(10):
                guUAVdistance[x]=(((gu_memory[0][x]-gu_memory[0][currentloc])**2)+((gu_memory[1][x]-gu_memory[1][currentloc])**2))**(1/2) 
            #find current UAV's location

            for x in range(10):
                if guUAVdistance[x] == 0:
                    currentloc=x
            if a==0:
                moveStack[stackcount][a]=currentloc
            print("distance, currentloc= "+str(currentloc))
            print(guUAVdistance)
            print("Sum of distance")
            print(guUAVdistanceSum)
            guUAVdistance[currentloc]=1000
            
            if b!=0 and a==b:
                if c==0:
                    nextloc=np.argmin(guUAVdistance)
                    guUAVdistance[nextloc]=1000
                else:
                    for x in range(c+1):
                        nextloc=np.argmin(guUAVdistance)
                        guUAVdistance[nextloc]=1000
            print(guUAVdistance)
            nextloc=np.argmin(guUAVdistance)
            guUAVdistanceSum+=guUAVdistance[np.argmin(guUAVdistance)]
            guUAVarray[a]=guUAVdistanceSum
            
            print("currentloc="+str(currentloc)+", nextloc="+str(nextloc))
            print("a="+str(a)+", b="+str(b)+", c="+str(c))
            
            gu_memory[0][currentloc]=150
            gu_memory[1][currentloc]=0
            currentloc=nextloc
            moveStack[stackcount][a+1]=currentloc
            if has_duplicates(moveStack[stackcount])==1:
                sumArray[stackcount]=1000
            else:
                sumArray[stackcount]=guUAVdistanceSum
        
        print("distances",end='')
        print(sumArray[stackcount])
        print("stacks",end='')
        print(moveStack[stackcount])
        stackcount+=1
"""
        #test
"""n=input('continue? ')
        if n=='a':
            continue
        else:
            quit()
"""
        #findloc end