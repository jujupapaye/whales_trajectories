from models_odenet import ODEFunc
import numpy as np 
import time
import torch
import torch.nn as nn
import torch.optim as optim 
import matplotlib.pyplot as plt
import dataset 
from transform_dataset import Rotate, AleatoryTranslate3D
import argparse

"""
Training of Neural Ordianary Differential Equations models
from data of cetaceans trajectories.
"""

parser = argparse.ArgumentParser(description='Training a NODE model with whales trajectories data.')
parser.add_argument('--dataset', type=str, choices=['toulon', '3Ddata'], default='toulon', help="choose dataset to train the model")
parser.add_argument('--hidden_size', type=int, default=150, help="hidden size of the NODE model")
parser.add_argument('--mode', type=str, choices=['normal', 'addp0', 'random'], default='random', help="method to train the model")
parser.add_argument('--epochs', type=int, default=200, help="number of epochs in the training")
parser.add_argument('--stoch_coeff', type=float, default=0.1, help="how random is the step for random method")
args = parser.parse_args()

gpu = 1
adjoint = False
if adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint




if __name__ == '__main__':
    n_iters = args.epochs
    add_p0 = args.mode != 'normal'
    if args.dataset=='toulon':
        trajectories = dataset.FinWhalesTrajectories2KM3(transform=Rotate(), add_P0=add_p0,normalizeXY=True, normalizeT=True)
        if add_p0:
            data_dim = 5  # x,y,t,x0,y0
        else:
            data_dim = 3 # x,y,t
    elif args.dataset=='3Ddata':
        trajectories = dataset.data_3D(transform=AleatoryTranslate3D(),add_p0=add_p0, normalizeXYZ=True,normalizeT=True)
        if add_p0:
            data_dim = 7  # x,y,z,t,x0,y0,z0
        else:
            data_dim = 4 # x,y,z,t
    if args.mode == 'random':
        coef = str(args.stoch_coeff)[0] + str(args.stoch_coeff)[2]
        name_model = 'model/odefunc_' + args.dataset + '_' + args.mode + coef + '_'+str(args.hidden_size)+'.pt'
    else:
        name_model = 'model/odefunc_' + args.dataset + '_' +args.mode + '_'+str(args.hidden_size)+'.pt'
    device = torch.device('cuda:' + str(gpu) if torch.cuda.is_available() else 'cpu')
    hidden_dim = args.hidden_size
    func = ODEFunc(device, data_dim=data_dim, hidden_dim=hidden_dim).to(device)
    optimizer = optim.RMSprop(func.parameters(), lr=1e-2)
    min_loss = 10
    for i in range(n_iters):
        end = time.time()
        loss_mean = 0
        for traj in trajectories:
            samp_traj, t  = traj['traj'].to(device), traj['t'].to(device)
            y0 = samp_traj[0,:].reshape((1,1,data_dim)).float().to(device)
            samp_traj = samp_traj.reshape((samp_traj.shape[0],1,1,data_dim)).float().to(device)
            samp_ts = t.float().to(device)
            optimizer.zero_grad()

            if args.mode=='random':
                pred_y = odeint(func, y0, samp_ts, method='stoch_rk4', options={'k': args.stoch_coeff})
            else:    
                pred_y = odeint(func, y0, samp_ts)
            loss = torch.mean(torch.abs(pred_y - samp_traj))
            loss.backward()
            optimizer.step()
            loss_mean += loss.item()
        loss_mean = loss_mean / len(trajectories)
        print("epoch / loss / time", i, loss_mean,"    ",time.time() - end)
        if loss_mean < min_loss:
            min_loss = loss_mean
            torch.save(func.state_dict(),name_model)