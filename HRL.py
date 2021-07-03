"""Homeostatic reinforcement learning.

Described in the paper: 
Continuous Homeostatic Reinforcement Learning 
for Self-Regulated Autonomous Agents.
"""

from config import Cfg_env, Cfg_agent, Cfg_nets
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import set_all_seeds

# Random seed
seed = 0
set_all_seeds(seed)


class Environment:
    def __init__(self, cfg: Cfg_env):
        self.cfg_env = cfg

    def is_point_inside(self, x, y):
        """Check if a point (x,y) is inside the polygon"""
        coords = self.cfg_env.coord_env
        lab = list(coords.keys())
        n_left = 0
        for i in range(0, len(lab) - 2, 2):
            if (((coords[lab[i + 1]] > y) and (coords[lab[i + 3]] <= y)) or ((coords[lab[i + 1]] <= y) and (coords[lab[i + 3]] > y))) and (coords[lab[i]] <= x):
                n_left += 1
        if (((coords[lab[len(lab) - 1]] > y) and (coords[lab[1]] <= y)) or ((coords[lab[len(lab) - 1]] <= y) and (coords[lab[1]] > y))) and (coords[lab[len(lab) - 1]] <= x):
            n_left += 1
        if n_left % 2 == 1:
            return True
        else:
            return False

    def is_segment_inside(self, xa, xb, ya, yb):
        """Check if the segment AB with A(xa, ya) and B(xb, yb) is completely inside
        the polygon"""
        coords = self.cfg_env.coord_env
        lab = list(coords.keys()) + list(coords.keys())[:2]
        n_inter = 0
        if (xa != xb):
            alpha_1 = (yb - ya) / (xb - xa)
            beta_1 = (ya * xb - yb * xa) / (xb - xa)
            for i in range(0, len(lab) - 2, 2):
                if coords[lab[i]] == coords[lab[i + 2]]:
                    inter = [coords[lab[i]], alpha_1 *
                             coords[lab[i]] + beta_1]
                    inter_in_AB = (inter[0] >= np.minimum(xa, xb)) and (inter[0] <= np.maximum(
                        xa, xb)) and (inter[1] >= np.minimum(ya, yb)) and (inter[1] <= np.maximum(ya, yb))
                    inter_in_border = (inter[0] >= np.minimum(coords[lab[i]], coords[lab[i + 2]])) and (inter[0] <= np.maximum(coords[lab[i]], coords[lab[i + 2]])) and (
                        inter[1] >= np.minimum(coords[lab[i + 1]], coords[lab[i + 3]])) and (inter[1] <= np.maximum(coords[lab[i + 1]], coords[lab[i + 3]]))
                    if inter_in_AB and inter_in_border:
                        n_inter += 1
                else:
                    if ya == yb:
                        if ya == coords[lab[i + 1]]:
                            if (np.minimum(xa, xb) <= np.maximum(coords[lab[i]], coords[lab[i + 2]])) and (np.maximum(xa, xb) >= np.minimum(coords[lab[i]], coords[lab[i + 2]])):
                                n_inter += 1
                    else:
                        inter = [(coords[lab[i + 1]] - beta_1) /
                                 alpha_1, coords[lab[i + 1]]]
                        inter_in_AB = (inter[0] >= np.minimum(xa, xb)) and (inter[0] <= np.maximum(
                            xa, xb)) and (inter[1] >= np.minimum(ya, yb)) and (inter[1] <= np.maximum(ya, yb))
                        inter_in_border = (inter[0] >= np.minimum(coords[lab[i]], coords[lab[i + 2]])) and (inter[0] <= np.maximum(coords[lab[i]], coords[lab[i + 2]])) and (
                            inter[1] >= np.minimum(coords[lab[i + 1]], coords[lab[i + 3]])) and (inter[1] <= np.maximum(coords[lab[i + 1]], coords[lab[i + 3]]))
                        if inter_in_AB and inter_in_border:
                            n_inter += 1
        else:
            for i in range(0, len(lab) - 2, 2):
                if coords[lab[i]] == coords[lab[i + 2]]:
                    if xa == coords[lab[i]]:
                        if (np.minimum(ya, yb) <= np.maximum(coords[lab[i + 1]], coords[lab[i + 3]])) and (np.maximum(ya, yb) >= np.minimum(coords[lab[i + 1]], coords[lab[i + 3]])):
                            n_inter += 1
                else:
                    inter = [xa, coords[lab[i + 1]]]
                    inter_in_AB = (inter[0] >= np.minimum(xa, xb)) and (inter[0] <= np.maximum(
                        xa, xb)) and (inter[1] >= np.minimum(ya, yb)) and (inter[1] <= np.maximum(ya, yb))
                    inter_in_border = (inter[0] >= np.minimum(coords[lab[i]], coords[lab[i + 2]])) and (inter[0] <= np.maximum(coords[lab[i]], coords[lab[i + 2]])) and (
                        inter[1] >= np.minimum(coords[lab[i + 1]], coords[lab[i + 3]])) and (inter[1] <= np.maximum(coords[lab[i + 1]], coords[lab[i + 3]]))
                    if inter_in_AB and inter_in_border:
                        n_inter += 1
        if n_inter > 0:
            return False
        else:
            return True

    def plot(self, ax=None, save_fig=False):
        """Plot the environment, but not the Agent.

        Parameters
        ----------
        ax: SubplotBase
        save_fig: bool

        Returns
        -------
        Returns nothing. Only inplace update ax.
        """
        coords = self.cfg_env.coord_env
        if ax is None:
            ax = plt.subplot(111)

        lab = list(coords.keys())
        for i in range(0, len(lab) - 2, 2):
            ax.plot([coords[lab[i]], coords[lab[i + 2]]],
                    [coords[lab[i + 1]], coords[lab[i + 3]]],
                    '-', color='black', lw=2)
        ax.plot([coords[lab[len(lab) - 2]], coords[lab[0]]],
                [coords[lab[len(lab) - 1]], coords[lab[1]]],
                '-', color='black', lw=2)

        for circle_name, circle in self.cfg_env.coord_circ.items():
            x = circle[0]
            y = circle[1]
            r = circle[2]
            color = circle[3]
            patch_circle = Circle((x, y), r, color=color)

            ax.add_patch(patch_circle)

            ax.text(x, y, circle_name)

        ax.axis('off')

        if save_fig:
            ax.savefig('environment.eps', format='eps')


class Agent:
    def __init__(self, x_star, c, angle_visual_field):
        """
        Initialize the Agent

        ...

        Variables
        ---------
        x_star: np.array
            homeostatic set point.
        c: np.array
            homogeneus to the inverse of a second. For example c = (-0.1, ...)
            says that the half-life (like a radioactive element) of the first
            ressource is equal to 10 seconds.
        angle_visual_field: float
            in radiant. Not implemented.
        """

        self.x_star = x_star
        self.zeta = np.zeros(9)
        self.zeta[6] = 3  # initialization position for the agent
        self.zeta[7] = 2  # initialization position for the agent
        self.c = c
        self.angle_visual_field = angle_visual_field

    def dynamics(self, zeta, u):
        """
        Return the Agent's dynamics which is represented by the f function.

        ...

        Variables
        ---------
        zeta: np.array
            whole world state.
        u: np.array
            control. (freewill of the agent)
        """
        f = np.zeros(zeta.shape)
        # Those first coordinate are homeostatic, and with a null control, zeta tends to zero.
        # coordinate 0 : resource 1
        # coordinate 1 : resource 2
        # coordinate 2 : resource 3
        # coordinate 3 : resource 4
        # coordinate 4 : muscular energy (muscular resource)
        # coordinate 6 : aware energy (aware resource) : low if sleepy.
        f[:6] = self.c[:6] * (zeta[:6] + self.x_star) + \
            u[:6] * (zeta[:6] + self.x_star)

        # Those coordinates are not homeostatic : they represent the x-speed,
        # y-speed, and angular-speed.
        # The agent can choose himself his speed.
        f[6:9] = u[6:9]
        return f

    def drive(self, zeta, epsilon=0.001):
        """
        Return the Agent's drive which is the distance between the agent's 
        state and the homeostatic set point.
        ...

        Variables
        ---------
        zeta: np.array
            whole world state.
        u: np.array
            control. (freewill of the agent)
        """
        # in the delta, we only count the internal state.
        # The tree last coordinate do not count in the homeostatic set point.
        delta = zeta[:6]
        drive_delta = np.sqrt(epsilon + np.dot(delta, delta))
        return drive_delta


class Net_J(nn.Module):
    """ Net_J is the tentative of the agent to learn the world.
    It is an approximation of the real J, which is the expected drive (-expected reward).
    The agents wants to minimize this drive.
    """

    def __init__(self, n_neurons, dropout_rate):
        super(Net_J, self).__init__()
        self.fc1 = nn.Linear(9, n_neurons)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(n_neurons, n_neurons)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(n_neurons, 1)

    def forward(self, x):
        """Return a real number. Not a vector"""
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        x = self.dropout1(x)
        x = self.fc3(x)
        output = torch.sigmoid(x)
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

    def __init__(self, n_neurons, dropout_rate):
        super(Net_f, self).__init__()
        # 9 is the size of zeta and 14 is the size of size of
        # the one-hot encoded action (the number of actions)

        # TODO: zeta is continuous and the one-hot-encoded control is kind of discrete.
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


cfg_env = Cfg_env()
cfg_agent = Cfg_agent()
cfg_nets = Cfg_nets()
cfg_algo = Cfg_algo()
cfg_actions = Cfg_actions()

env = Environment(cfg_env)
agent = Agent(cfg_agent)

net_J = Net_J(cfg_nets)
net_f = Net_f(cfg_nets)

algo = Algorithm(env, agent, net_J, net_f,
                 cfg_algo, cfg_actions)


algo.simulation()
