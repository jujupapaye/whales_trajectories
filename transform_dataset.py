import numpy as np
import torch

class Rotate(object):
    """Rotate (between 0 and 5 degrees) a trajectory around a random point of the trajectory"""
    def __call__(self, sample):
            traj, t, specy = sample['traj'], sample['t'], sample['specy']
            random_index = np.random.randint(len(traj))
            ox, oy = traj[random_index,0], traj[random_index,1]
            degrees = np.random.uniform(5)
            angle = degrees * np.pi / 180  # in radians
            rotate_traj = torch.empty_like(traj).copy_(traj)
            for i in range(traj.shape[0]):
                if i != random_index:
                    px, py = traj[i,0], traj[i,1]
                    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
                    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
                    rotate_traj[i,0] = qx
                    rotate_traj[i,1] = qy
            return {'traj': rotate_traj, 't': t, 'specy': specy}

class AlignTrajectories(object):
    """Translation, then rotate to put all trajectories on the same axes, begining with (0,0)"""
    def __call__(self, sample):
            traj, t, specy = sample['traj'], sample['t'], sample['specy']

            translate_traj = torch.empty_like(traj).copy_(traj)
            x0, y0 = traj[0,0] ,traj[0,1]  # first point of trajectory
            for i in range(traj.shape[0]):
                translate_traj[i,0] = traj[i,0] - x0
                translate_traj[i,1] = traj[i,1] - y0
                translate_traj[i,2] = t[i]
            
            ox, oy = translate_traj[0,0], translate_traj[0,1]  # center is (0,0) = first point of trajectory
            xt, yt = translate_traj[-1,0] ,translate_traj[-1,1]  # last point of trajectory
            a = np.abs(yt) / np.abs(xt)  # pente de la ligne droite entre premier point et dernier point de la traj
            
            if xt <= 0 and yt <= 0:
                angle = -np.arctan(np.abs(yt) / np.abs(xt)) + np.pi  # in radians
            elif xt > 0 and yt <= 0:
                angle = np.arctan(np.abs(yt) / np.abs(xt))
            elif xt <= 0 and yt > 0:
                angle = np.arctan(np.abs(yt) / np.abs(xt)) 
            elif xt > 0 and yt > 0:
                angle = -np.arctan(np.abs(yt) / np.abs(xt))
            rotate_traj = torch.empty_like(translate_traj).copy_(translate_traj)
            rotate_traj[0,0], rotate_traj[0,1] = 0, 0
            for i in range(1, traj.shape[0]):
                px, py = translate_traj[i,0], translate_traj[i,1]
                qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
                qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
                rotate_traj[i,0] = qx
                rotate_traj[i,1] = qy
            return {'traj': rotate_traj, 't': t, 'specy': specy}


class TranslateRotateTrajectories(object):
    """Translation to (0,0), then aleatory rotation of the point of the trajectory"""
    def __call__(self, sample):
            traj, t, specy = sample['traj'], sample['t'], sample['specy']

            translate_traj = torch.empty_like(traj).copy_(traj)
            x0, y0 = traj[0,0] ,traj[0,1]  # first point of trajectory
            for i in range(traj.shape[0]):
                translate_traj[i,0] = traj[i,0] - x0
                translate_traj[i,1] = traj[i,1] - y0
                translate_traj[i,2] = t[i]
            
            ox, oy = translate_traj[0,0], translate_traj[0,1]  # center is (0,0) = first point of trajectory
            xt, yt = translate_traj[-1,0] ,translate_traj[-1,1]  # last point of trajectory
            angle = np.random.randint(0,360)*np.pi/180
            rotate_traj = torch.empty_like(translate_traj).copy_(translate_traj)
            rotate_traj[0,0], rotate_traj[0,1] = 0, 0
            for i in range(1, traj.shape[0]):
                px, py = translate_traj[i,0], translate_traj[i,1]
                qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
                qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
                rotate_traj[i,0] = qx 
                rotate_traj[i,1] = qy 
            return {'traj': rotate_traj, 't': t, 'specy': specy}


class TranslateTrajectories(object):
    """Translation to (0,0) of all the point of the trajectory"""
    def __call__(self, sample):
            traj, t , specy = sample['traj'].squeeze(), sample['t'], sample['specy']
            translate_traj = torch.empty_like(traj).copy_(traj)
            x0, y0 = traj[0,0] ,traj[0,1]  # first point of trajectory
            for i in range(traj.shape[0]):
                translate_traj[i,0] = (traj[i,0] - x0) 
                translate_traj[i,1] = (traj[i,1] - y0)
                translate_traj[i,2] = t[i] 
            return {'traj': translate_traj, 't': t , 'specy' : specy}