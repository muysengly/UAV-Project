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

sxpos=10
trial = int(X_MAX/10)-1
arrayXsize=int(X_MAX/10)
arrayYsize=int(Y_MAX/10)
arrayZsize=7
groupArea=np.zeros((arrayXsize,arrayYsize,arrayZsize))+11

#initial group search
for x in range(trial):
    sypos=10
    for y in range(trial):
        count=0
        print("x="+str(sxpos)+", y="+str(sypos))
        for i in range(10):
            if sxpos-MAX_BEAM_RADIUS<=gu_memory[0][i]<=sxpos+MAX_BEAM_RADIUS and sypos-MAX_BEAM_RADIUS<=gu_memory[1][i]<=sypos+MAX_BEAM_RADIUS:
                print(str(i)+", ",end='')
                a=int(sxpos/10)-1
                b=int(sypos/10)-1
                groupArea[a][b][count]=i
                count+=1
                print(", a="+str(a)+" b="+str(b)+" count="+str(count))
        print()
        plt.scatter(x=sxpos, y=sypos, c="red")
        plt.text(x=sxpos, y=sypos - 4, s=f"{len(np.unique(groupArea[int(sxpos/10)-1][int(sypos/10)-1]))-1}")
        sypos+=10
    sxpos+=10   
groupArea_size=np.zeros((9,9))
for i in range(9):
    for j in range(9):
        groupArea_size[i][j]=len(np.unique(groupArea[i][j]))-1
#print(len(np.unique(groupArea[8][0]))-1)
print(groupArea_size)

groupArea_maxSize=np.max(groupArea_size) #size of biggest group

#find x,y location of biggest group points
#first max start
firstMax_size=0
for i in range(9):
    firstMax_size+=list(groupArea_size[i]).count(groupArea_maxSize)
print(firstMax_size)

firstMax=np.zeros((firstMax_size,2))
firstMax_counter=0

for i in range(9):
    for j in range(9):
        if groupArea_size[i][j]==groupArea_maxSize:
            firstMax[firstMax_counter][0]=int(i)
            firstMax[firstMax_counter][1]=int(j)
            firstMax_counter+=1
print(firstMax)
maxGUlist=np.zeros(9)+11
print(firstMax_size)
print(firstMax_counter)
print(groupArea_maxSize)
for i in range(firstMax_counter):
    x1=int(firstMax[i][0])
    y1=int(firstMax[i][1])
    len1 = len(np.unique(np.concatenate([maxGUlist,groupArea[x1][y1]])))
    counter4=0
    print("previous:"+str(maxGUlist)+", "+str(groupArea[x1][y1])+" = "+str((np.unique(np.concatenate([maxGUlist,groupArea[x1][y1]])))))
    tmpList=np.unique(np.concatenate([maxGUlist,groupArea[x1][y1]]))
    for j in range(len1-1):
        print(maxGUlist)
        maxGUlist[j]=tmpList[j]
        counter4+=1
     
        
print(maxGUlist)
for i in range(len(np.unique(maxGUlist))-1):
    for x in range(arrayXsize):
        for y in range(arrayYsize):
            for z in range(arrayZsize):
                if groupArea[x][y][z]==maxGUlist[i] and (len(np.unique(groupArea[x][y]))-1)!=groupArea_maxSize:
                    groupArea[x][y]=11
print(groupArea)
        
#first max end


"""n=input('continue? '+str(sxpos)+', '+str(sypos+10))
        if n=='a':
            continue
        else:
            quit()"""

#makeBeamCirclewDot(uavspot[count2][0],uavspot[count2][1],MAX_BEAM_DIAMETER,'skyblue')


 



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