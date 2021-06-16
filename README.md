# Whales trajectories processing with Neural Ordinary Differential Equations

## trajectory.py 

It contains tools for processing trajectory data.

## dataset.py

It contains all dataset builder. Every dataset is a trajectory database. See details are in the code.

Every sample of a database is a dictionary with 3 components {'traj': traj, 't': t, 'specy': specy} :
- traj : an array of shape (nb_of_points, 3) where all point are longitude, latitude, t
    or an array of shape (nb_of_points, 5) where all point are longitude, latitude, t, long0, lat0 if add_P0=True
- t : time in seconds from the point of departure of the traj
- specy : 'fin whales' or 'blue whales' or 'sperm whales"


FinWhalesDayTrajectoriesKM3() : Trajectories per day of fin whales dataset in Toulon (KM3 data)

FinWhalesTrajectories2KM3() : Trajectories of fin whales dataset in Toulon (KM3 data), the trajectories are made if there are more than 2 hours between 2 points.

FinAndBlueWhalesCalifornia(): Trajectory per animals of the california dataset

FinAndBlueWhalesCaliforniaLittleTraj(): Trajectory per animals of the california dataset cutting by number of points

FinAndBlueWhalesCaliforniaTimedTraj(): Trajectory per animals of the california dataset cutting by duration

data_3D() : 15 trajectories of 3D locations of sprem whales off the coast of Antibes 

All constructors have these options (default: False): 
- add_p0=True : add the point of departure to every point. The data size then increases by 2 (or 3 for 3D data). And every point is representing as : (x,y,t,x0,y0)
- normalizeXY=True : normalize the coordinates of the trajectory between 0 and 1
- normalizeT=True : normalize the time t between 0 and 1 (advised)



## transform_dataset.py

It contains all transformations that we can applied to trajectories data. (rotation, translation...)

## models_odenet.py

It contains all necesary to construct some Neural Ordinary Differential Equations models and some normal networks:
- ODEFunc : 3 linear layer with tanh non linearity
- ODENet : Odefunc mapping + 1 linear layer to do regression or classification
- ConvODEFunc : convolutional ode function
- ConvODENet : ConvODEFunc block + 1 linear layer to do regression or classification
and more.

## SCRIPT THAT WE CAN EXECUTE

### train.py 
usage : train.py [-h] [--dataset {toulon,3D_data}] [--hidden_size HIDDEN_SIZE] [--mode {normal,add_p0,random}] [--epochs EPOCHS] [--stoch_coeff STOCH_COEFF]

Used to train and build a NODE model from trajectory dataset (Toulon's fin whales or sperm whales 3D). The constructed model is an ODEFunc as defined in models_odenet.py. We can parametrized the hidden_size in option (default 150). The number of epochs can be chosen too (default 200). 

Parameter --mode:
- normal :
- add_p0 :
- random :

The learned function is then saved in model/odefunc_ .pt where ... 

Once the model is learned and save, we can generate trajectories and the corresponding vector plot with generate_trajectories.py, and generate_vector_plot.py respectively.

## generate_trajectories.py
usage : generate_trajectories.py [--model {odefunc.pt}] [--nb NB] [--t T]

Generate and show NB trajectories from odefunc.pt model learned previously with train.py

## generate_vector_plot.py
usage : generate_vector_plot.py [--model {odefunc.pt}]

Generate and show the corresponding vector plot from the odefunc.pt function learned previously with train.py.

