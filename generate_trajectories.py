import matplotlib.pyplot as plt
import numpy as np
import torch 
from models_odenet import ODEFunc
import argparse

parser = argparse.ArgumentParser(description='Training a NODE model with whales trajectories data.')
parser.add_argument('--model', type=str, default='model/odefunc_toulon_normal_150.pt', help="the name of the model learned previously with train.py")
parser.add_argument('--nb_traj', type=int, default=20, help="number of simulated trajectories that we want to simulate")
parser.add_argument('--gpu', type=int, default=1, help="gpu number")
args = parser.parse_args()

adjoint = False
if adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


if __name__ == '__main__':
    arguments = args.model.split('_')
    dataset = arguments[1]
    mode = arguments[2]
    hidden_dim = int(arguments[3][:-3])
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    if dataset=='toulon':
        data_dim = 3
        if mode=='addp0' or mode[:-2]=='random':
            data_dim += 2
    else:
        data_dim = 4
        if mode=='addp0' or mode[:-2]=='random':
            data_dim += 3
    func = ODEFunc(device, data_dim=data_dim, hidden_dim=hidden_dim, augment_dim=0).to(device)
    func.load_state_dict(torch.load(args.model))
    nb_traj = args.nb_traj
    departures = np.random.uniform(0,1,size=(nb_traj,data_dim))
    #departures = np.ones((nb_traj,data_dim))*0.2

    if data_dim == 5:
        departures[:,3] = departures[:,0]
        departures[:,4] = departures[:,1]
    if data_dim == 7:
        departures[:,4] = departures[:,0]
        departures[:,5] = departures[:,1]
        departures[:,6] = departures[:,2]

    if dataset == 'toulon':
        fig, ax = plt.subplots()
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    for y0 in departures:
        l = np.random.uniform(0,1)
        t = torch.linspace(0, l ,100).to(device)
        y0 = torch.from_numpy(y0).float().to(device)
        if mode[:-2] == 'random':  # random mode
            stoch_coeff = float(mode[-2] + '.' + mode[-1])
            pred_y = odeint(func, y0, t, method='stoch_rk4', options={'k': stoch_coeff})
        else:
            pred_y = odeint(func, y0, t)
        pred_y = pred_y.cpu().squeeze().detach().numpy()
        if dataset == 'toulon':
            plt.plot(pred_y[:,0], pred_y[:,1], alpha=0.9)
            plt.scatter(pred_y[0,0],pred_y[0,1],marker='x')
        else:  # 3D data
            ax.plot(pred_y[:,0], pred_y[:,1], pred_y[:,2], alpha=0.9)
            ax.scatter(pred_y[0,0],pred_y[0,1],pred_y[0,2],marker='x')
    if dataset == 'toulon':
        plt.scatter(0.3617, 0.3617, label='hydrophone',color='black')
    if dataset=='toulon':
        plt.xlabel('normalized longitude')
        plt.ylabel('normalized latitude')
        plt.title('Simulation of fin whales trajectories in 2D')
    else:  # 3D dataset
        plt.xlabel('x')
        plt.ylabel('y')
        #plt.zlabel('z')
        plt.title('Simulation of sperm whales trajectories in 3D')
    plt.grid()
    plt.legend()
    plt.show()