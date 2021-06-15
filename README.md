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



## transform_dataset.py

It contains all transformations that we can applied to trajectories data. (rotation, translation...)

## generate_trajectory.py

## models_odenet.py and models.py

It contains all necesary to construct ...

## SCRIPT THAT WE CAN EXECUTE

### run_dynamics.py

### run_with_p0.py

### run_adding_random.py

### run_latent_ode.py


