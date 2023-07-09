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

#find distance of gu-UAV
count12=1
nextloc=0
guUAVdistanceSum=0
guUAVarray=np.zeros(10)
sumArray=np.zeros(56)
moveStack=np.zeros((56,10))
stackcount=0

#loop start
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
        #test
        """n=input('continue? ')
        if n=='a':
            continue
        else:
            quit()
"""
        #findloc end

#print("distance: "+str(round(guUAVdistanceSum,2))+"m")
print("distances",end='')
print(sumArray)
print("stacks",end='')
print(moveStack)
attemptNo=np.argmin(sumArray)
print(sumArray[attemptNo])
print(moveStack[attemptNo])
for x in range(2):
    for y in range(10):
        gu_memory[x][y]=int(df.iloc[y+1,x])

for x in range(9):
    yy = int(moveStack[attemptNo][x])
    yyy=int(moveStack[attemptNo][x+1])
    plt.plot([gu_memory[0][yy],gu_memory[0][yyy]],[gu_memory[1][yy],gu_memory[1][yyy]],color="red")

wb=xl.Workbook()
sheet1= wb.active
sheet1.title='stack'
for x in range(10):
    sheet1.cell(row=1+x,column=1,value=moveStack[attemptNo][x])
wb.save(filename='stacks.xlsx')


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