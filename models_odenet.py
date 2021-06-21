import torch
import torch.nn as nn
from torchdiffeq import odeint
from torchdiffeq import odeint_adjoint


MAX_NUM_STEPS = 1000  # Maximum number of steps for ODE solver

####################################
#        ODE FUNC AND ODENET       #
####################################

class ODEFunc(nn.Module):
    def __init__(self, device, data_dim, hidden_dim, augment_dim=0):
        super(ODEFunc, self).__init__()
        self.device = device
        self.data_dim = data_dim
        self.input_dim = data_dim + augment_dim
        self.hidden_dim = hidden_dim
        self.augment_dim = augment_dim
        self.nfe = 0  # Number of function evaluations
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, self.input_dim)
        self.non_linearity = nn.Tanh()

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = self.non_linearity(out)
        out = self.fc2(out)
        out = self.non_linearity(out)
        out = self.fc3(out)
        return out


class ODEBlock(nn.Module):
    def __init__(self, device, odefunc, is_conv=False, tol=1e-3, adjoint=False):
        super(ODEBlock, self).__init__()
        self.adjoint = adjoint
        self.device = device
        self.is_conv = is_conv
        self.odefunc = odefunc
        self.tol = tol

    def forward(self, x, eval_times=None):
        self.odefunc.nfe = 0
        if eval_times is None:
            integration_time = torch.tensor([0, 1]).float().type_as(x)
        else:
            integration_time = eval_times.type_as(x)

        if self.odefunc.augment_dim > 0:
            aug = torch.zeros(x.shape[0], self.odefunc.augment_dim).to(self.device)
            x = torch.cat([x, aug], 1)

        if self.adjoint:
            out = odeint_adjoint(self.odefunc, x, integration_time,
                                 rtol=self.tol, atol=self.tol, method='dopri5',
                                 options={'max_num_steps': MAX_NUM_STEPS})
        else:
            out = odeint(self.odefunc, x, integration_time,
                         rtol=self.tol, atol=self.tol, method='dopri5',
                         options={'max_num_steps': MAX_NUM_STEPS})

        if eval_times is None:
            return out[1]  # Return only final time
        else:
            return out

    def trajectory(self, x, timesteps):
        integration_time = torch.linspace(0., 1., timesteps)
        return self.forward(x, eval_times=integration_time)



class ODENet(nn.Module):
    def __init__(self, device, data_dim, hidden_dim, output_dim=1, augment_dim=0,
                 tol=1e-3, adjoint=False):
        super(ODENet, self).__init__()
        self.device = device
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.tol = tol

        odefunc = ODEFunc(device, data_dim, hidden_dim, augment_dim)

        self.odeblock = ODEBlock(device, odefunc, tol=tol, adjoint=adjoint)
        self.linear_layer = nn.Linear(self.odeblock.odefunc.input_dim, self.output_dim)

    def forward(self, x, return_features=False):
        features = self.odeblock(x)
        pred = self.linear_layer(features)
        if return_features:
            return features, pred
        return pred

class LatentODEfunc(nn.Module):
    def __init__(self, latent_dim=4, nhidden=20):
        super(LatentODEfunc, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, latent_dim)
        self.nfe = 0
        #self.elu = nn.Softplus()

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc3(out)
        return out


class RecognitionRNN(nn.Module):
    def __init__(self, latent_dim=4, obs_dim=2, nhidden=25, nbatch=1):
        super(RecognitionRNN, self).__init__()
        self.nhidden = nhidden
        self.nbatch = nbatch
        self.i2h = nn.Linear(obs_dim + nhidden, nhidden)
        self.h2o = nn.Linear(nhidden, latent_dim * 2)

    def forward(self, x, h):
        combined = torch.cat((x, h), dim=1)
        h = torch.tanh(self.i2h(combined))
        out = self.h2o(h)
        return out, h

    def initHidden(self):
        return torch.zeros(self.nbatch, self.nhidden)


class Decoder(nn.Module):
    def __init__(self, latent_dim=4, obs_dim=2, nhidden=20):
        super(Decoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, obs_dim)

    def forward(self, z):
        out = self.fc1(z)
        out = self.relu(out)
        out = self.fc2(out)
        return out

####################################
#        CONV ODE NET              #
####################################

class Conv2dTime(nn.Conv2d):
    """
    Implements time dependent 2d convolutions, by appending the time variable as
    an extra channel.
    """
    def __init__(self, in_channels, *args, **kwargs):
        super(Conv2dTime, self).__init__(in_channels + 1, *args, **kwargs)

    def forward(self, t, x):
        # Shape (batch_size, 1, height, width)
        t_img = torch.ones_like(x[:, :1, :, :]) * t
        # Shape (batch_size, channels + 1, height, width)
        t_and_x = torch.cat([t_img, x], 1)
        return super(Conv2dTime, self).forward(t_and_x)


class ConvODEFunc(nn.Module):
    """Convolutional block modeling the derivative of ODE system.
    Parameters
    ----------
    device : torch.device
    img_size : tuple of ints
        Tuple of (channels, height, width).
    num_filters : int
        Number of convolutional filters.
    augment_dim: int
        Number of augmentation channels to add. If 0 does not augment ODE.
    time_dependent : bool
        If True adds time as input, making ODE time dependent.
    non_linearity : string
        One of 'relu' and 'softplus'
    """
    def __init__(self, device, img_size, num_filters, augment_dim=0,
                 time_dependent=False, non_linearity='relu'):
        super(ConvODEFunc, self).__init__()
        self.device = device
        self.augment_dim = augment_dim
        self.img_size = img_size
        self.time_dependent = time_dependent
        self.nfe = 0  # Number of function evaluations
        self.channels, self.height, self.width = img_size
        self.channels += augment_dim
        self.num_filters = num_filters

        if time_dependent:
            self.conv1 = Conv2dTime(self.channels, self.num_filters,
                                    kernel_size=1, stride=1, padding=0)
            self.conv2 = Conv2dTime(self.num_filters, self.num_filters,
                                    kernel_size=3, stride=1, padding=1)
            self.conv3 = Conv2dTime(self.num_filters, self.channels,
                                    kernel_size=1, stride=1, padding=0)
        else:
            self.conv1 = nn.Conv2d(self.channels, self.num_filters,
                                   kernel_size=1, stride=1, padding=0)
            self.conv2 = nn.Conv2d(self.num_filters, self.num_filters,
                                   kernel_size=3, stride=1, padding=1)
            self.conv3 = nn.Conv2d(self.num_filters, self.channels,
                                   kernel_size=1, stride=1, padding=0)

        if non_linearity == 'relu':
            self.non_linearity = nn.ReLU(inplace=True)
        elif non_linearity == 'softplus':
            self.non_linearity = nn.Softplus()

    def forward(self, t, x):
        """
        Parameters
        ----------
        t : torch.Tensor
            Current time.
        x : torch.Tensor
            Shape (batch_size, input_dim)
        """
        self.nfe += 1
        if self.time_dependent:
            out = self.conv1(t, x)
            out = self.non_linearity(out)
            out = self.conv2(t, out)
            out = self.non_linearity(out)
            out = self.conv3(t, out)
        else:
            #print(x.shape)
            out = self.conv1(x)
            #print(out.shape)
            out = self.non_linearity(out)
            out = self.conv2(out)
            #print(out.shape)
            out = self.non_linearity(out)
            out = self.conv3(out)
            #print(out.shape)
            #exit(0)
        return out


class ConvODENet(nn.Module):
    """Creates an ODEBlock with a convolutional ODEFunc followed by a Linear
    layer.
    Parameters
    ----------
    device : torch.device
    img_size : tuple of ints
        Tuple of (channels, height, width).
    num_filters : int
        Number of convolutional filters.
    output_dim : int
        Dimension of output after hidden layer. Should be 1 for regression or
        num_classes for classification.
    augment_dim: int
        Number of augmentation channels to add. If 0 does not augment ODE.
    time_dependent : bool
        If True adds time as input, making ODE time dependent.
    non_linearity : string
        One of 'relu' and 'softplus'
    tol : float
        Error tolerance.
    adjoint : bool
        If True calculates gradient with adjoint method, otherwise
        backpropagates directly through operations of ODE solver.
    """
    def __init__(self, device, img_size, num_filters, output_dim=1,
                 augment_dim=0, time_dependent=False, non_linearity='relu',
                 tol=1e-3, adjoint=False):
        super(ConvODENet, self).__init__()
        self.device = device
        self.img_size = img_size
        self.num_filters = num_filters
        self.augment_dim = augment_dim
        self.output_dim = output_dim
        self.flattened_dim = (img_size[0] + augment_dim) * img_size[1] * img_size[2]
        self.time_dependent = time_dependent
        self.tol = tol

        odefunc = ConvODEFunc(device, img_size, num_filters, augment_dim,
                              time_dependent, non_linearity)

        self.odeblock = ODEBlock(device, odefunc, is_conv=True, tol=tol,
                                 adjoint=adjoint)

        self.linear_layer = nn.Linear(self.flattened_dim, self.output_dim)

    def forward(self, x, return_features=False):
        features = self.odeblock(x)
        pred = self.linear_layer(features.view(features.size(0), -1))
        if return_features:
            return features, pred
        return pred

####################################
#        SIMPLE NEURAL NETWORKS    #
####################################


class SimpleNet(nn.Module):
    def __init__(self, device, data_dim, hidden_dim, output_dim=1):
        super(SimpleNet, self).__init__()
        self.device = device
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(self.data_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim*2)
        self.fc3 = nn.Linear(hidden_dim*2, hidden_dim*3)
        self.fc4 = nn.Linear(hidden_dim*3, self.output_dim)
        self.non_linearity = nn.ReLU() #nn.Tanh()

    def forward(self,x):
        out = self.fc1(x)
        out = self.non_linearity(out)
        out = self.fc2(out)
        out = self.non_linearity(out)
        out = self.fc3(out)
        out = self.non_linearity(out)
        out = self.fc4(out)
        return out

class CNN(nn.Module):
    def __init__(self, device, data_dim, output_dim=1):
        super(SimpleNet, self).__init__()
        self.height, self.width = img_size
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(64, self.num_filters,kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, self.channels,kernel_size=1, stride=1, padding=0)
        self.non_linearity = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(16,self.outputdim)

    def forward(self,x):
        out = self.conv1(x)
        out = self.non_linearity(out)
        out = self.conv2(out)
        out = self.non_linearity(out)
        out = self.conv3(out)
        print(out.shape)
        out = self.flatten(out)
        out = self.linear(out)
        return x



####################################
#        CDE MODEL                 #
####################################

######################
# A CDE model looks like
#
# z_t = z_0 + \int_0^t f_\theta(z_s) dX_s
#
# Where X is your data and f_\theta is a neural network. So the first thing we need to do is define such an f_\theta.
# That's what this CDEFunc class does.
# Here we've built a small single-hidden-layer neural network, whose hidden layer is of width 128.
######################
class CDEFunc(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels):
        ######################
        # input_channels is the number of input channels in the data X. (Determined by the data.)
        # hidden_channels is the number of channels for z_t. (Determined by you!)
        ######################
        super(CDEFunc, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.linear1 = torch.nn.Linear(hidden_channels, 128)
        self.linear2 = torch.nn.Linear(128, input_channels * hidden_channels)

    ######################
    # For most purposes the t argument can probably be ignored; unless you want your CDE to behave differently at
    # different times, which would be unusual. But it's there if you need it!
    ######################
    def forward(self, t, z):
        # z has shape (batch, hidden_channels)
        z = self.linear1(z)
        z = z.relu()
        z = self.linear2(z)
        ######################
        # Easy-to-forget gotcha: Best results tend to be obtained by adding a final tanh nonlinearity.
        ######################
        z = z.tanh()
        ######################
        # Ignoring the batch dimension, the shape of the output tensor must be a matrix,
        # because we need it to represent a linear map from R^input_channels to R^hidden_channels.
        ######################
        z = z.view(z.size(0), self.hidden_channels, self.input_channels)
        return z


######################
# Next, we need to package CDEFunc up into a model that computes the integral.
######################
class NeuralCDE(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, interpolation="cubic"):
        super(NeuralCDE, self).__init__()

        self.func = CDEFunc(input_channels, hidden_channels)
        self.initial = torch.nn.Linear(input_channels, hidden_channels)
        self.readout = torch.nn.Linear(hidden_channels, output_channels)
        self.interpolation = interpolation

    def forward(self, coeffs):
        if self.interpolation == 'cubic':
            X = torchcde.NaturalCubicSpline(coeffs)
        elif self.interpolation == 'linear':
            X = torchcde.LinearInterpolation(coeffs)
        else:
            raise ValueError("Only 'linear' and 'cubic' interpolation methods are implemented.")

        ######################
        # Easy to forget gotcha: Initial hidden state should be a function of the first observation.
        ######################
        X0 = X.evaluate(X.interval[0])
        z0 = self.initial(X0)

        ######################
        # Actually solve the CDE.
        ######################
        z_T = torchcde.cdeint(X=X,
                              z0=z0,
                              func=self.func,
                              t=X.interval)

        ######################
        # Both the initial value and the terminal value are returned from cdeint; extract just the terminal value,
        # and then apply a linear map.
        ######################
        z_T = z_T[:, 1]
        pred_y = self.readout(z_T)
        return pred_y
