import torch.nn as nn
import torch.nn.functional as F
from torch import sigmoid


class Net_J(nn.Module):
    """ Net_J is the tentative of the agent to learn the world.
    It is an approximation of the real J, which is the expected 
    drive (-expected reward).
    The agents wants to minimize this drive.
    """

    def __init__(self):
        super(Net_J, self).__init__()
        n_neurons = 128
        dropout_rate = 0.15
        self.fc1 = nn.Linear(9, n_neurons)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(n_neurons, n_neurons)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(n_neurons, 1)

    def forward(self, x):
        """Return a real number. Not a vector"""
        x = self.fc1(x)
        x = sigmoid(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = sigmoid(x)
        x = self.dropout1(x)
        x = self.fc3(x)
        output = sigmoid(x)
        return output


class Net_f(nn.Module):
    """ f is homogeneous to the speed of change of Zeta.
    For example the rate of consumption of glucose
    or the rate of consumption of water

    d_zeta = f((zeta, u)) * dt.
    Net_f is the tentative of the agent to modelise f.

    Zeta is all there is. It is the whole world
    zeta = internal + external state
    """

    def __init__(self):
        super(Net_f, self).__init__()
        n_neurons = 128
        dropout_rate = 0.15

        # 9 is the size of zeta and 14 is the size of size of
        # the one-hot encoded action (the number of actions)

        # TODO: zeta is continuous and the one-hot-encoded control
        # is kind of discrete.
        # We could preprocess zeta before concatenating with the control.
        self.fc1 = nn.Linear(9 + 14, n_neurons)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(n_neurons, n_neurons)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(n_neurons, 9)

    def forward(self, x):
        """Return a speed homogeneous to zeta."""
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout1(x)
        output = self.fc3(x)
        return output
