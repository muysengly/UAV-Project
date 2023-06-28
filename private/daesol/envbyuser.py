import numpy as np

def showMapSky(environment,howlong):
    for i in range(howlong):
        for j in range(howlong):
            print(environment[i][j],end='')
        print()

def showMapGnd(environment,world,howlong):
    for i in range(howlong):
        for j in range(howlong):
            print(environment[world][i][j],end='')
        print()

def showMapGnd1(environment,first,second):
    z=first*3+second
    for i in range(5):
        for j in range(5):
            print(environment[z][i][j],end='')
        print()

def showMap(character,first,second):
    sky[first][second] = character
    print("sky map:")
    showMapSky(sky,3)
    print("ground map:")
    showMapGnd1(gnd,first,second)

def showUavStatus(uavbattery,reward):
    print("UAV battery: ",end='')
    print(uavbattery)
    print("Reward: ",end='')
    print(reward)

def control1(x,y,i,j,uavBattery,batteryPerGrid,uav):
    sky[x][y]=0
    x=x+i
    y=y+j
    uavBattery-=batteryPerGrid
    showMap(uav,x,y)

    

#ground map start
gnd = np.ones((9,5,5),dtype="U4")
for k in range(9):
    for i in range(5):
        for j in range(5):
            gnd[k][i][j]="x"
#0-00, 1-01, 2-02, 3-10, 4-11, 5-12, 6-20, 7-21, 8-22
gnd[0][1][2]=0
gnd[0][2][2]=0
gnd[3][1][2]=0
gnd[4][2][2]="%"
gnd[5][2][4]=0
gnd[7][3][1]=0
gnd[8][2][2]=0

#ground map end
#sky map start
sky=np.ones((3,3),dtype="U4")
for i in range(3):
    for j in range(3):
        sky[i][j]=0
#sky map end
#UAV status
uav = "@"
x=1
y=1
uavBattery=100
batteryPerGrid=15
#UAV status end
#rewards
reward=0
rewardList=[]
searchReward=np.ones(9)
searchReward-=searchReward
searched=np.ones(9)
searched=searchReward
agentEpisode=50
moveStack=[[],[],[]]
#rewards end
showMap(uav,x,y)
#for epi in range(agentEpisode):
while True:
    #1

    if uavBattery<30:
        print("go to charge station")
    a=input("")
    if a=="z":
        break
    elif a=="a":
        if 0<y:
            sky[x][y]=0
            y-=1
            uavBattery-=batteryPerGrid
            showMap(uav,x,y)
        else:
            continue

    elif a=="d":
        if y<2:
            sky[x][y]=0
            y+=1
            uavBattery-=batteryPerGrid
            showMap(uav,x,y)
        else:
            continue
    
    elif a=="s":
        if x<2:
            sky[x][y]=0
            x+=1
            uavBattery-=batteryPerGrid
            showMap(uav,x,y)
        else:
            continue
    
    elif a=="w":
        if 0<x:
            sky[x][y]=0
            x-=1
            uavBattery-=batteryPerGrid
            showMap(uav,x,y)
        else:
            continue  
    if searched[x*3+y]>0:
        reward-=2
    for i in range(5):
        for j in range(5):
            if gnd[x*3+y][i][j] == "0":
                gnd[x*3+y][i][j]="0f"
                searchReward[x*3+y]+=1
                reward+=5
                uavBattery-=5
                if searched[x*3+y]<1:
                    searched[x*3+y]+=1
            else:
                if searched[x*3+y]<1:
                    searched[x*3+y]+=1
    #print(searchReward[x*3+y])
    for i in range(5):
        for j in range(5):
            if gnd[x*3+y][i][j] == "%":
                uavBattery=100
    if uavBattery<=0:
        reward-=10
        print("no more battery")
        rewardList.append(reward)
        break
    if np.sum(searchReward)==6:
        reward+=30
        print("found all")
        rewardList.append(reward)
        break
    showUavStatus(uavBattery,reward)


        


    


