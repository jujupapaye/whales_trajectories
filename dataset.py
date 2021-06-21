import pandas as pd
from trajectory import get_traj2, get_trajectories_per_days
from torch.utils.data import Dataset
import torch
import numpy as np
import random
from transform_dataset import Rotate, TranslateRotateTrajectories, AlignTrajectories,TranslateTrajectories


class FinWhalesDayTrajectoriesKM3(Dataset):
    """Trajectories per day of fin whales dataset in Toulon (KM3 data)
    Every sample is a dictionary with 3 components:
    - traj : an array of shape (nb_of_points, 3) where all point are x, y, t
    - t : time in seconds from the point of departure of the traj
    - specy : 'fin whales'
    """

    def __init__(self, transform=None, normalizeXY=False, normalizeT=False):
        self.trajectories = get_trajectories_per_days()
        self.tort_values = []
        self.transform = transform
        self.normalizeXY = normalizeXY
        self.normalizeT = normalizeT
        self.specy = 'fin whales'
        self.max_T = 85548
        self.min_XY = -18132
        self.max_XY = 31994
        if normalizeXY:
            self.norm = lambda x : (x-self.min_XY)/(self.max_XY-self.min_XY) 
        else:
            self.norm = lambda x : x

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        traj = traj.sort_values(by=['date'])
        t = np.zeros((len(traj)))
        first = traj['date'][traj.index[0]]
        for i in range(1, len(traj)):
            dt = traj['date'][traj.index[i]] - first
            if self.normalizeT:
                t[i] = ( dt.seconds + dt.days*24*3600 ) / self.max_T
            else:
                t[i] = ( dt.seconds + dt.days*24*3600 )
        x0, y0 = self.norm(traj['x'][traj.index[0]]), self.norm(traj['y'][traj.index[0]])
        coord = np.array([self.norm(np.array([traj['x']])), self.norm(np.array([traj['y']])), np.array([t])], dtype=float).T
        traj = torch.from_numpy(coord.reshape((coord.shape[0],3)))
        samp_t = torch.from_numpy(t)
        if self.transform is not None:
            return self.transform({'traj': traj, 't': samp_t, 'specy': self.specy})
        return {'traj': traj, 't': samp_t, 'specy': self.specy}


class FinWhalesTrajectories2KM3(Dataset):
    """Trajectories of fin whales dataset in Toulon (KM3 data)
    the trajectories are made if there are more than 2 hours between 2 points.

    Every sample is a dictionary with 3 components:
    - traj : an array of shape (nb_of_points, 3) where all point are x, y, t
    or an array of shape (nb_of_points, 5) where all point are x, y, t, x0, y0 if add_P0=True
    - t : time in seconds from the point of departure of the traj
    - specy : 'fin whales'
    """

    def __init__(self, transform=None, add_P0=False,normalizeXY=False, normalizeT=False):
        self.trajectories = get_traj2()
        self.tort_values = []
        self.transform = transform
        self.add_P0 = add_P0
        self.normalizeXY = normalizeXY
        self.normalizeT = normalizeT
        self.specy = 'fin whales'
        self.max_T = 30290
        self.min_XY = -18132
        self.max_XY = 31994
        if normalizeXY:
            self.norm = lambda x : (x-self.min_XY)/(self.max_XY-self.min_XY) 
        else:
            self.norm = lambda x : x

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        traj = traj.sort_values(by=['date'])
        t = np.zeros((len(traj)))
        first = traj['date'][traj.index[0]]
        for i in range(1, len(traj)):
            dt = traj['date'][traj.index[i]] - first
            if self.normalizeT:
                t[i] = ( dt.seconds + dt.days*24*3600 ) / self.max_T
            else:
                t[i] = ( dt.seconds + dt.days*24*3600 )
        x0, y0 = self.norm(traj['x'][traj.index[0]]), self.norm(traj['y'][traj.index[0]])
        if self.add_P0:
            coord = np.array([self.norm(np.array([traj['x']])), self.norm(np.array([traj['y']])), np.array([t]),self.norm(np.full((1,len(traj)) ,x0)), self.norm(np.full((1,len(traj)) , y0))], dtype=float).T
            traj = torch.from_numpy(coord.reshape((coord.shape[0],5))).float()
        else:
            coord = np.array([self.norm(np.array([traj['x']])), self.norm(np.array([traj['y']])), np.array([t])], dtype=float).T
            traj = torch.from_numpy(coord.reshape((coord.shape[0],3))).float()
        samp_t = torch.from_numpy(t).float()
        if self.transform is not None:
            return self.transform({'traj': traj, 't': samp_t, 'specy': self.specy})
        return {'traj': traj, 't': samp_t, 'specy': self.specy}



class FinAndBlueWhalesCalifornia(Dataset):
    """Data from Irvine LM, Winsor MH, Follett TM, Mate BR, Palacios DM (2020) 
    An at-sea assessment of Argos location accuracy for 2 species of large whales, 
    and the effect of deep-diving behavior on location error. 
    Animal Biotelemetry 8:20. doi:10.1186/s40317-020-00207-x

    Trajectory per animals of the california dataset (fin whales or bblue whales).

    PARAMETERS
    specy ='fin' to get fin whales dataset (2081 positions) or 'blue' to get blue whales dataset (15088 positions)
    normalizeXY=True normalize the coordinates of the trajectory between 0 and 1.
    normalizeT=True normalize the time t between 0 and 1.
    add_p0=True add at every point of the trajectory the point of departure of the traj.

    Every sample is a dictionary with 3 components:
    - traj : an array of shape (nb_of_points, 3) where all point are longitude, latitude, t
    or an array of shape (nb_of_points, 5) where all point are longitude, latitude, t,long0, lat0 if add_P0=True
    - t : time in seconds from the point of departure of the traj
    - specy : 'fin whales' or 'blue whales'
    """
 
    def __init__(self, specy='fin', transform=None, normalizeXY=False, normalizeT=False):
        self.path = './data/california3.csv'
        self.transform = transform
        self.normalizeXY = normalizeXY
        self.normalizeT = normalizeT
        data = pd.read_csv(self.path)
        data = data[['timestamp', 'location-long', 'location-lat','individual-taxon-canonical-name','individual-local-identifier']]
        data.columns = ['t','long','lat','specy','name']
        data = data[data['long'].isnull()==False]  # on enlève les Nan
        if specy == 'fin':  # fin whales dataset
            self.specy = 'fin whales'
            rorqual = data[data['specy']=='Balaenoptera physalus']
            indexes_ror = rorqual.index
            traj_ror = []
            for r in rorqual['name'].unique():  # for all different fin whales
                traj_ror.append(rorqual[rorqual['name']==r].sort_values(by=['t']))
            self.trajectories = traj_ror

        else:  # blue whales dataset
            self.specy = 'blue whales'
            baleine = data[data['specy']=='Balaenoptera musculus']
            indexes_bal = baleine.index
            traj_bal = []
            for b in baleine['name'].unique():  # for all different blue whales
                traj_bal.append(baleine[baleine['name']==b].sort_values(by=['t']))
            self.trajectories = traj_bal
    
    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        coordinate = torch.zeros((len(traj), 3))  # long, lat, t
        time = torch.zeros(len(traj))  # t = time in minutes
        indexes = traj.index
        first = pd.to_datetime(traj['t'][indexes[0]])
        coordinate[0,:] = torch.Tensor([traj['long'][indexes[0]], traj['lat'][indexes[0]], time[0]])
        for i in range(1,len(traj)):
            time[i] = (pd.to_datetime(traj['t'][indexes[i]]) - first).seconds/60 + (pd.to_datetime(traj['t'][indexes[i]]) - first).days * 24 * 60
            coordinate[i,:] = torch.Tensor([traj['long'][indexes[i]], traj['lat'][indexes[i]], time[i]])
        return {'traj': coordinate, 't': time, 'specy': self.specy}


class FinAndBlueWhalesCaliforniaLittleTraj(Dataset):
    """Dataset of fin OR blue whales from the california dataset trajectories
    ,cutting by number of points
    
    PARAMETERS
    duration: duration of a trajectory in number of points
    stride: stride to cut the trajectory
    normalizeXY=True normalize the coordinates of the trajectory between 0 and 1.
    normalizeT=True normalize the time t between 0 and 1
    add_p0=True add at every point of the trajectory the point of departure of the traj.

    Every sample is a dictionary with 3 components:
    - traj : an array of shape (nb_of_points, 3) where all point are longitude, latitude, t
    or an array of shape (nb_of_points, 5) where all point are longitude, latitude, t, long0, lat0 if add_P0=True
    - t : time in seconds from the point of departure of the traj
    - specy : 'fin whales' or 'blue whales'
    """
    def __init__(self, specy='fin', duration=30, stride=20, transform=None, add_p0=False, normalizeXY=False,normalizeT=False):  
        self.duration = duration
        self.stride = stride
        self.specy = 'fin whales' if specy=='fin' else 'blue whales' 
        self.transform = transform
        self.normalizeXY = normalizeXY
        self.normalizeT = normalizeT
        self.max_dt = 0  # maximum length in time of a trajectory
        if specy == 'fin':  # fin whales dataset
            fin = FinAndBlueWhalesCalifornia('fin')
            self.traj_per_animal = fin.trajectories
        else:  # blue whales dataset
            blue = FinAndBlueWhalesCalifornia('blue')
            self.traj_per_animal = blue.trajectories
        trajectories = []
        for traj in self.traj_per_animal:
            indexes = traj.index
            if len(indexes) > self.duration:
                for i in range(0, len(indexes)-self.duration, self.stride):
                    new_traj = traj.loc[indexes[i:i+self.duration], traj.columns]
                    dt = (pd.to_datetime(new_traj['t'][new_traj.index[-1]]) - pd.to_datetime(new_traj['t'][new_traj.index[0]]) )
                    if (dt.days*24*3600 + dt.seconds) > self.max_dt:
                        self.max_dt = dt.days*24*3600 + dt.seconds
                    trajectories.append(new_traj)
            else:
                dt = (pd.to_datetime(traj['t'][traj.index[-1]]) - pd.to_datetime(traj['t'][traj.index[0]]) )
                if (dt.days*24*3600 + dt.seconds) > self.max_dt:
                    self.max_dt = dt.days*24*3600 + dt.seconds
                trajectories.append(traj)
        self.trajectories = trajectories
        self.min_XY = -131.4611
        self.max_XY = 40.4511
        if normalizeXY:
            self.norm = lambda x : (x-self.min_XY)/(self.max_XY-self.min_XY) 
        else:
            self.norm = lambda x : x

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        coordinate = torch.zeros((len(traj), 3))  # long, lat, t
        time = torch.zeros(len(traj))  # t
        indexes = traj.index
        first = pd.to_datetime(traj['t'][indexes[0]])
        coordinate[0,:] = torch.Tensor([self.norm(traj['long'][indexes[0]]), self.norm(traj['lat'][indexes[0]]), time[0]])
        for i in range(1,len(traj)):
            dt = (pd.to_datetime(traj['t'][indexes[i]]) - first)
            if self.normalizeT:
                time[i] = (dt.days*24*60*60 + dt.seconds) / self.max_dt  
            else:
                time[i] = (dt.days*24*60*60 + dt.seconds)
            coordinate[i,:] = torch.Tensor([self.norm(traj['long'][indexes[i]]), self.norm(traj['lat'][indexes[i]]), time[i]])
        if self.transform is not None:
            return self.transform({'traj': coordinate, 't': time})
        return {'traj': coordinate, 't': time, 'specy': self.specy}

class FinAndBlueWhalesCaliforniaTimedTraj(Dataset):
    """Dataset of fin OR blue whales from the california dataset trajectories
    ,cutting in time.
    
    PARAMETERS
    duration: duration MAXIMUM of a trajectory in minutes
    stride: stride to cut the trajectory
    normalizeXY=True normalize the coordinates of the trajectory between 0 and 1.
    normalizeT=True normalize the time t between 0 and 1
    add_p0=True add at every point of the trajectory the point of departure of the traj.

    Every sample is a dictionary with 3 components:
    - traj : an array of shape (nb_of_points, 3) where all point are longitude, latitude, t
    or an array of shape (nb_of_points, 5) where all point are longitude, latitude, t, long0, lat0 if add_P0=True
    - t : time in seconds from the point of departure of the traj
    - specy : 'fin whales' or 'blue whales'
    """
    def __init__(self, specy='fin', duration=180, stride=10, transform=None, add_p0=False, normalizeXY=False,normalizeT=False):  
        self.duration = duration   # in minutes
        self.stride = stride
        self.specy = 'fin whales' if specy=='fin' else 'blue whales' 
        self.transform = transform
        self.normalizeXY = normalizeXY
        self.normalizeT = normalizeT
        self.add_p0 = add_p0
        if specy == 'fin':  # fin whales dataset
            fin = FinAndBlueWhalesCalifornia('fin')
            self.traj_per_animal = fin.trajectories
        else:  # blue whales dataset
            blue = FinAndBlueWhalesCalifornia('blue')
            self.traj_per_animal = blue.trajectories
        trajectories = []
        for traj in self.traj_per_animal:
            indexes = traj.index
            for i in range(0, len(indexes), self.stride):
                begin = traj.loc[indexes[i]]
                dt = 0
                for j in range(i, len(indexes)):
                    end = traj.loc[indexes[j], traj.columns]
                    dt = (pd.to_datetime(end['t']) - pd.to_datetime(begin['t']) )
                    if (dt.days*24*60 + dt.seconds/60) > self.duration:
                        new_traj = traj.loc[indexes[i:j-1], traj.columns]
                        if len(new_traj) > 2:
                            trajectories.append(new_traj)
                        break
        self.trajectories = trajectories
        self.min_XY = -131.4611
        self.max_XY = 40.4511
        if normalizeXY:
            self.norm = lambda x : (x-self.min_XY)/(self.max_XY-self.min_XY) 
        else:
            self.norm = lambda x : x

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        time = torch.zeros(len(traj))  # t
        indexes = traj.index
        first = pd.to_datetime(traj['t'][indexes[0]])
        if self.add_p0:
            coordinate = torch.zeros((len(traj), 5))  # long, lat, t, long0, lat0
            coordinate[0,:] = torch.Tensor([self.norm(traj['long'][indexes[0]]), self.norm(traj['lat'][indexes[0]]), time[0], self.norm(traj['long'][indexes[0]]), self.norm(traj['lat'][indexes[0]])])
        else:
            coordinate = torch.zeros((len(traj), 3))  # long, lat, t
            coordinate[0,:] = torch.Tensor([self.norm(traj['long'][indexes[0]]), self.norm(traj['lat'][indexes[0]]), time[0]])
        for i in range(1,len(traj)):
            dt = (pd.to_datetime(traj['t'][indexes[i]]) - first)
            if self.normalizeT:
                time[i] = (dt.days*24*60*60 + dt.seconds) / (self.duration*60)  
            else:
                time[i] = dt.days*24*60*60 + dt.seconds
            if self.add_p0:
                coordinate[i,:] = torch.Tensor([self.norm(traj['long'][indexes[i]]), self.norm(traj['lat'][indexes[i]]), time[i], self.norm(traj['long'][indexes[0]]), self.norm(traj['lat'][indexes[0]])])
            else:
                coordinate[i,:] = torch.Tensor([self.norm(traj['long'][indexes[i]]), self.norm(traj['lat'][indexes[i]]), time[i]])
        if self.transform is not None:
            return self.transform({'traj': coordinate, 't': time})
        return {'traj': coordinate, 't': time, 'specy': self.specy}





class Fin_And_BlueWhalesCalifornia(Dataset):
    """Dataset of a mix of fin whales and blue whales from the data of California trajectory 
    from Irvine and her team (useful for clasification)"""
    def __init__(self, duration=30, stride=20, transform=None, add_p0=False, normalizeXYZ=False,normalizeT=False):  
        self.duration = duration
        self.stride = stride
        self.specy = 'fin and blue whales'
        self.transform = transform
        self.max_dt = 0  # maximum length in time of a trajectory
        fin = FinAndBlueWhalesCalifornia('fin').trajectories
        blue = FinAndBlueWhalesCalifornia('blue').trajectories
        self.traj_per_animal = fin + blue
        trajectories = []
        for traj in self.traj_per_animal:
            indexes = traj.index
            if len(indexes) > self.duration:
                for i in range(0, len(indexes)-self.duration, self.stride):
                    new_traj = traj.loc[indexes[i:i+self.duration], traj.columns]
                    dt = (pd.to_datetime(new_traj['t'][new_traj.index[-1]]) - pd.to_datetime(new_traj['t'][new_traj.index[0]]) )
                    if (dt.days*24*60 + dt.seconds/60) > self.max_dt:
                        self.max_dt = dt.days*24*60 + dt.seconds/60
                    trajectories.append(new_traj)
            else:
                dt = (pd.to_datetime(traj['t'][traj.index[-1]]) - pd.to_datetime(traj['t'][traj.index[0]]) )
                if (dt.days*24*60 + dt.seconds/60) > self.max_dt:
                    self.max_dt = dt.days*24*60 + dt.seconds/60
                trajectories.append(traj)
        blue = trajectories[101:] 
        random.shuffle(blue)
        self.trajectories = trajectories[0:101] + blue[0:101]

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        coordinate = torch.zeros((self.duration, 3))  # long, lat, t
        time = torch.zeros(self.duration)  # t
        indexes = traj.index
        first = pd.to_datetime(traj['t'][indexes[0]])
        coordinate[0,:] = torch.Tensor([traj['long'][indexes[0]], traj['lat'][indexes[0]], 0])
        for i in range(1,len(traj)):
            dt = (pd.to_datetime(traj['t'][indexes[i]]) - first)
            time[i] = (dt.days*24*60 + dt.seconds/60) / self.max_dt  # in minutes 
            coordinate[i,:] = torch.Tensor([traj['long'][indexes[i]], traj['lat'][indexes[i]], time[i]])
        specy = torch.zeros((2))
        if traj['specy'][traj.index[0]]=='Balaenoptera musculus':  # blue whales
            specy[0] = 1
        else: # fin whales
            specy[1] = 1
        if self.transform is not None:
            return self.transform({'traj': coordinate, 't': time, 'specy' : specy})
        return {'traj': coordinate, 't': time, 'specy' : specy}



class data_3D(Dataset):
    """Dataset of 3D (x,y,z,t) locations of sperm whales respected to time.
    
    PARAMETERS
    normalizeXYZ=True normalize the coordinates of the trajectory between 0 and 1.
    normalizeT=True normalize the time t between 0 and 1
    add_p0=True add at every point of the trajectory the point of departure of the traj.

    Every sample is a dictionary with 3 components:
    - traj : an array of shape (nb_of_points, 4) where all point are x,y,z,t
    or an array of shape (nb_of_points, 7) where all point are x,y,z,t,x0,y0,t0 if add_p0=True
    - t : time in seconds from the point of departure of the traj
    - specy : 'sperm whales'
    """
    def __init__(self, transform=None, add_p0=False, normalizeXYZ=False,normalizeT=False):  
        self.transform = transform
        folder = "data/record_14/"
        self.trajectories = []
        self.add_p0 = add_p0
        self.normalizeXYZ = normalizeXYZ
        self.normalizeT = normalizeT
        for i in range(1,16):
            traj = pd.read_csv(folder + "record_14_01_track" + str(i) + ".csv")
            self.trajectories.append(traj[traj['X'].isnull() == False])  # remove Nan Data
        self.max_dt = 2310  # duration maximum of a trajectory in seconds (used to normalized t)
        self.max_XYZ = 1253 # max of all x,y,z
        self.min_XYZ = -3831 # min of all x,y,z
        if self.normalizeXYZ:
            self.norm = lambda x : (x-self.min_XYZ)/(self.max_XYZ-self.min_XYZ) 
        else:
            self.norm = lambda x : x

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        if self.add_p0:
            coordinate = torch.zeros((len(traj), 7))  # X, Y, Z, t, X0, Y0, t0
        else:
            coordinate = torch.zeros((len(traj), 4))  # X, Y, Z, t
        time = torch.zeros(len(traj))  # t
        indexes = traj.index
        first = traj['Sec'][indexes[0]]+ traj['Min'][indexes[0]]*60 + traj['Hour'][indexes[0]]*3600
        if self.add_p0:
            coordinate[0,:] = torch.Tensor([self.norm(traj['X'][indexes[0]]), self.norm(traj['Y'][indexes[0]]), self.norm(traj['Z'][indexes[0]]), 0, self.norm(traj['X'][indexes[0]]), self.norm(traj['Y'][indexes[0]]), self.norm(traj['Z'][indexes[0]])])
        else:
            coordinate[0,:] = torch.Tensor([self.norm(traj['X'][indexes[0]]), self.norm(traj['Y'][indexes[0]]), self.norm(traj['Z'][indexes[0]]), 0])
        for i in range(1,len(traj)):
            dt = traj['Sec'][indexes[i]]+ traj['Min'][indexes[i]]*60 + traj['Hour'][indexes[i]]*3600 - first
            if self.normalizeT:
                time[i] = dt/self.max_dt  # in normalized secondes (between 0 and 1)
            else:
                time[i] = dt  # in secondes
            if self.add_p0:
                coordinate[i,:] = torch.Tensor([self.norm(traj['X'][indexes[i]]), self.norm(traj['Y'][indexes[i]]), self.norm(traj['Z'][indexes[i]]), time[i], self.norm(traj['X'][indexes[0]]), self.norm(traj['Y'][indexes[0]]), self.norm(traj['Z'][indexes[0]])])
            else:
                coordinate[i,:] = torch.Tensor([self.norm(traj['X'][indexes[i]]) , self.norm(traj['Y'][indexes[i]]), self.norm(traj['Z'][indexes[i]]), time[i]])
        if self.transform is not None:
            return self.transform({'traj': coordinate, 't': time, 'specy': 'sperm whales'})
        return {'traj': coordinate, 't': time, 'specy': 'sperm whales'}