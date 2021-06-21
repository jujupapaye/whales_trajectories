import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from get_data import open_data
from torch.utils.data import TensorDataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
import matplotlib.cm as cm

# compute euclidean distance between (xA, yA) and (xB, yB)
def distance(xA, yA, xB, yB):
    return np.sqrt((xB-xA)**2+(yB-yA)**2)

# normalization with the map boundaries
def normalize(x):
    return (x+18132)/(31994+18132)

# compute turtosity of the trajectory traj
def turtosity(traj):
    bird_flight_dist = distance(traj[0,0], traj[0,1], traj[-1,0], traj[-1,1]) 
    dist_traj = 0
    x0, y0 = traj[0,0], traj[0,1]
    for p in traj:
        x1, y1 = p[0], p[1]
        dist_traj += distance(x0, y0, x1, y1)
        x0, y0 = x1, y1
    return dist_traj / bird_flight_dist 

# get a list of trajectories per days like [day1, ..., day12] from toulon's
# where dayi is a dataframe (12 trajectories)
def get_trajectories_per_days():
    nb_per_day = pd.read_csv('data/nb_per_day.csv')
    days = nb_per_day['date']
    data = open_data()
    trajectories = []
    for day in range(len(days)):
        j2 = days[day]
        if day > 0:
            j1 = days[day-1]
            traj_day = data[ (data['date'] > j1) & (data['date'] <= j2)]
        else:
            traj_day = data[data['date'] < j2]
        if len(traj_day)>3:
            trajectories.append(traj_day)
    return trajectories


# return a list of dtime, ddist, coupure where:
# dtime = time interval between 2 points that followed each other (Timedelta)
# ddist = distance interval between 2 points that followed each other (meter)
# coupure = where index to cut to do create trajectories
# the parameter traj define the condition to cut
def where_cutting_traj(data, traj=2):
    data_sorted = data.sort_values(by=['date'])
    indexes = data_sorted.index
    dtime = [] 
    ddist = []  
    for i in range(len(indexes)-1):
        t1, t2 = indexes[i], indexes[i+1]
        dt = data_sorted['date'][t2] - data_sorted['date'][t1]
        dtime.append(dt)
        dist = distance(data_sorted['x'][t1], data_sorted['y'][t1], data_sorted['x'][t2], data_sorted['y'][t2])
        ddist.append(dist)
    coupure = []
    for t,d,j in zip(dtime, ddist, np.arange(len(dtime))):
        if traj == 3:
            if t.days>0 or ((t.seconds > 60) and (d > 830)):  # plus d'une minute et plus de 830m (plus de 50km/h)
                coupure.append(j)
        else:
            if t.days>0 or t.seconds > 60*60*2:  # plus de 2h entre 2 pulses
                coupure.append(j+1)
    return dtime, ddist, coupure


# get a list of trajectories from the data w.r.t the cutting "coupure" made previouly with the where_cutting_traj function
def couper(data, coupure):
    data_sorted = data.sort_values(by=['date'])
    indexes = data_sorted.index
    trajectories = []
    begin = 0
    for cut in coupure:
        traj = data_sorted[begin:cut]
        if len(traj) > 3:
            trajectories.append(traj)
        begin = cut
    return trajectories

# get trajectories where trajectories are made is 2 points are separated with more than 2 hours of time
def get_traj2():
    data = open_data()
    dtime, ddist, coupure = where_cutting_traj(data)
    trajectories = couper(data, coupure)
    return trajectories

def get_traj3():
    data = open_data()
    dtime, ddist, coupure = where_cutting_traj(data, traj=3)
    trajectories = couper(data, coupure)
    return trajectories

# get 2 array p0, p1 where p1[i] corresponds to the point that followed p0[i] in a trajectory
def get_couples():
    data = open_data()
    trajectories = get_traj2()
    p0, p1 = [],[]
    for traj in trajectories:
        indexes = traj.index 
        for p in range(len(traj)-1):
            dt = traj['date'][indexes[p+1]] - traj['date'][indexes[p]]
            p0.append([traj['long'][indexes[p]], traj['lat'][indexes[p]] ,dt.seconds ]) 
            p1.append([traj['long'][indexes[p+1]], traj['lat'][indexes[p+1]] ,dt.seconds ])
    return np.array(p0), np.array(p1)

# get 2 array p0, p1 like in get_couples within a dataLoader
def get_couples_dataLoader(norm=True,batch_size=16, test_batch_size=32):
    x, y = get_couples()
    if norm:
        x = normalize(x)
        y = normalize(y)
    x = torch.Tensor(x)
    y = torch.Tensor(y)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    train_data = TensorDataset(X_train,y_train)
    train_loader = DataLoader(train_data, batch_size=batch_size)
    test_data = TensorDataset(X_test,y_test)
    test_loader = DataLoader(test_data, batch_size=test_batch_size)
    return train_loader, test_loader


# plot a vector field of the data of Toulon
def speed_vector_field():
    begin, end = get_couples()
    d = []
    for p0, p1 in zip(begin,end):
        dist = distance(p0[0], p0[1], p1[0], p1[1])
        d.append(dist)
    d = np.array(d)
    colors = d #/ begin[:,2] # linear mean speed
    X = begin[:, 0]
    Y = begin[:, 1]
    U = end[:, 0]- begin[:, 0]
    V = end[:, 1]- begin[:, 1]
    fig, ax = plt.subplots()
    q = ax.quiver(X, Y, U, V ,colors, angles='xy',scale_units='xy', scale=5,cmap='gist_ncar')
    plt.scatter(6.027565,42.806225,s=10,c='black', label='hydrophone')
    plt.clim(min(colors),max(colors))
    plt.colorbar()
    ax.grid()
    ax.set_aspect('equal')
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.legend()
    plt.show()







    