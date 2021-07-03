"""Homeostatic reinforcement learning.

Described in the paper: 
Continuous Homeostatic Reinforcement Learning 
for Self-Regulated Autonomous Agents.
"""

from config import Cfg_env, Cfg_agent, Cfg_nets, Cfg_algo, Cfg_actions
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import Normalize
import pandas as pd
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


class Algorithm:
    def __init__(self, env, agent, net_J, net_f,
                 time_step, eps, gamma, tau, N_iter, cycle_plot, N_rolling, N_save_weights, N_print, learning_rate,
                 actions_controls, constraints, min_time_sleep, max_tired, asym_coeff, min_resource, nb_actions):
        """
        """

        self.env = env
        self.agent = agent
        self.net_J = net_J
        self.net_f = net_f

        self.time_step = time_step
        self.eps = eps  # probability to choose an action ramdomly.
        self.gamma = gamma  # The discount factor.
        self.tau = tau  # the smoothing update parameter. Not implemented.
        self.N_iter = N_iter
        self.cycle_plot = cycle_plot
        self.N_rolling = N_rolling  # rolling average

        # Periodic saving of the weights of J and f.
        self.N_save_weights = N_save_weights
        self.N_print = N_print  # Periodic print of the zeta.
        self.learning_rate = learning_rate
        self.optimizer_J = torch.optim.Adam(
            self.net_J.parameters(), lr=self.learning_rate)
        self.optimizer_f = torch.optim.Adam(
            self.net_f.parameters(), lr=self.learning_rate)

        # a list mapping the action to a control.
        self.actions_controls = actions_controls
        # a vector representing the physical limits. Example: you cannot eat more than 6 kg of food...
        self.constraints = constraints
        # An agent cannot do micro sleep. He cannot wake during a minimum amount of time.
        self.min_time_sleep = min_time_sleep
        # If the agent is too tired, he automatically sleeps.
        self.max_tired = max_tired

        # coeffiscient to make the loss and thus the gradient larger  for small actions
        self.asym_coeff = asym_coeff

        # If one of the agent resource is lower than min_resource, we put it back at min_resource
        # This help because if one of the agent resource equals 0, because of the dynamics
        #  of the exponintial,
        # the agent cannot reconstitute this resource.
        self.min_resource = min_resource

        self.nb_actions = nb_actions

        self.historic_zeta = []
        self.historic_actions = []
        self.historic_losses = []  # will contain a list of 2d [L_f, L_J]

    def actions_possible(self):
        """
        Return a list of bool showing which action is permitted or not.

        possible_actions = [
            # walking
            # running
            # turning an angle to the left
            # turning an angle to the right
            # sleeping
            # get resource 1
            # get resource 2
            # get resource 3
            # get resource 4
            # not doing anything
        ]
        + 4 for the action of going to a resource after seeing it
        """
        # There are 4 more possible actions:
        # The 4 last actions correspond to going direcly to each of the 4 ressources if possible.
        possible_actions = [True for i in range(
            len(self.actions_controls) + 4)]

        # If the agent is too tired in his muscle
        if self.agent.zeta[4] >= self.constraints[0]:
            # walking is no more possible
            possible_actions[0] = False

        # He cannot escape the environment when walking.
        x_walk = self.agent.zeta[6] + self.time_step * \
            self.actions_controls[0][int(self.agent.zeta[8])][6]
        y_walk = self.agent.zeta[7] + self.time_step * \
            self.actions_controls[0][int(self.agent.zeta[8])][7]
        if not self.env.is_point_inside(x_walk, y_walk):
            possible_actions[0] = False

        # if the agent is too tired, we cannot run.
        if self.agent.zeta[4] >= self.constraints[1]:
            possible_actions[1] = False

        # He cannot escape the environment when running.
        x_run = self.agent.zeta[6] + self.time_step * \
            self.actions_controls[1][int(self.agent.zeta[8])][6]
        y_run = self.agent.zeta[7] + self.time_step * \
            self.actions_controls[1][int(self.agent.zeta[8])][7]

        if not self.env.is_point_inside(x_run, y_run):
            possible_actions[1] = False

        # He cannot rotate trigonometrically if too tired.
        if self.agent.zeta[4] >= self.constraints[2]:
            possible_actions[2] = False

        # He cannot rotate non-trigonometrically if too tired.
        if self.agent.zeta[4] >= self.constraints[3]:
            possible_actions[3] = False

        # The agent cannot sleep if he is not enouth tired.
        if self.agent.zeta[5] <= self.constraints[4]:
            possible_actions[4] = False

        # If the agent is too sleepy, the only possible action is to sleep.
        if self.agent.zeta[5] >= self.max_tired:
            possible_actions = [False for i in range(
                len(self.actions_controls) + 4)]
            possible_actions[4] = True

        def is_near_ressource(circle):
            """circle= (str) 'circle_1'"""
            dist = (self.agent.zeta[6] - self.env.coord_circ[circle][0])**2 + (
                self.agent.zeta[7] - self.env.coord_circ[circle][1])**2
            radius = self.env.coord_circ[circle][2]**2
            return dist < radius

        def check_resource(i):
            index_circle = 4+i
            if self.agent.zeta[0+i - 1] >= self.constraints[index_circle]:
                possible_actions[index_circle] = False
            if not is_near_ressource(f'circle_{str(i)}'):
                possible_actions[index_circle] = False

        for resource in range(1, 5):
            check_resource(resource)

        def is_resource_visible(resource):
            """Check if segment between agent and resource i is visible"""
            xa = self.agent.zeta[6]
            xb = self.env.coord_circ[f'circle_{str(resource)}'][0]
            ya = self.agent.zeta[7]
            yb = self.env.coord_circ[f'circle_{str(resource)}'][1]
            return self.env.is_segment_inside(xa, xb, ya, yb)

        for resource in range(1, 5):
            if not is_resource_visible(resource):
                possible_actions[9+resource] = False

        return possible_actions

    def new_state(self, a):
        """Return the new state after an action is taken.

        Parameter:
        ----------
        a: action.
        actions = [
            0# walking
            1# running
            2# turning an angle to the left
            3# turning an angle to the right
            4# sleeping
            5# get resource 1
            6# get resource 2
            7# get resource 3
            8# get resource 4
            9# not doing anything
        ]


        Returns:
        --------
        The new states.
        """
        if a == 4:
            # The new state is when the agent wakes up
            # Therefore, we integrate the differential equation until this time
            new_zeta = self.agent.zeta
            for i in range(6):
                new_zeta[i] = -self.agent.x_star[i] + (new_zeta[i] + self.agent.x_star[i]) * np.exp(
                    (self.agent.c[i] + self.actions_controls[4][i]) * self.min_time_sleep * self.time_step)
            return new_zeta

        action_circle = {
            10: "circle_1",
            11: "circle_2",
            12: "circle_3",
            13: "circle_4",
        }

        def going_and_get_resource(circle):
            """Return the new state associated with the special action a going 
            direclty to the circle.

            Parameter:
            ----------
            circle : example : "circle_1

            Returns:
            --------
            The new state (zeta), but with the agent who has wlaken to go to 
            the state and so which is therefore more tired.
            """
            new_zeta = self.agent.zeta
            d = np.sqrt((self.agent.zeta[6] - self.env.coord_circ[circle][0])**2 + (
                self.agent.zeta[7] - self.env.coord_circ[circle][1])**2)
            if d != 0:
                # If the agent is at a distance d from the resource, it will first need to walk
                # to consume it. Thus, we integrate the differential equation of its internal
                # state during this time
                t = d * self.time_step / 0.1
                for i in range(6):
                    new_zeta[i] = -self.agent.x_star[i] + (new_zeta[i] + self.agent.x_star[i]) * np.exp(
                        (self.agent.c[i] + self.actions_controls[0][0][i]) * t)
                new_zeta[6] = self.env.coord_circ[circle][0]
                new_zeta[7] = self.env.coord_circ[circle][1]
                return new_zeta
            else:
                # If the agent is already on the resource, then consuming it is done instantly
                u = self.actions_controls[9]
                new_zeta = self.agent.zeta + self.time_step * \
                    self.agent.dynamics(self.agent.zeta, u)
                return new_zeta

        for a_, circle in action_circle.items():
            if a == a_:
                new_zeta = going_and_get_resource(circle)

        else:
            u = np.zeros(self.agent.zeta.shape)

            # walk
            if a == 0:
                u = self.actions_controls[0][int(self.agent.zeta[8])]

            # run
            elif a == 1:
                u = self.actions_controls[1][int(self.agent.zeta[8])]

            # if other, we just select the control associated with this action.
            elif (a == 2) or (a == 3) or (a == 5) or (a == 6) or (a == 7) or (a == 8) or (a == 9):
                u = self.actions_controls[a]

            # Euler method to calculate the new zeta.
            new_zeta = self.agent.zeta + self.time_step * \
                self.agent.dynamics(self.agent.zeta, u)

            # If the agent is turning its angle
            # Not in euler because discretized.
            if a == 2:
                new_zeta[8] = new_zeta[8] + 1
            elif a == 3:
                new_zeta[8] = new_zeta[8] - 1
            if new_zeta[8] == len(self.actions_controls[0]):
                new_zeta[8] = 0
            elif new_zeta[8] == -1:
                new_zeta[8] = len(self.actions_controls[0]) - 1

            return new_zeta

    def evaluate_action(self, action):
        """Return the score associated with the action.

        In this function, we do not seek to update the Net_F and Net_J, so we use the eval mode.
        But we still seek to use the derivative of the Net_F according to zeta. So we use require_grad = True.
        Generally, inly the parameters of a neural network are on require_grad = True.
        But here we must use zeta.require_grad = True.

        Parameters:
        ----------
        action : int

        Returns:
        --------
        the score : pytorch float.
        """
        # f is a neural network taking one vector.
        # But this vector contains the information of zeta and u.
        # The u is the one-hot-encoded control associated with the action a
        zeta_u = np.concatenate(
            [self.agent.zeta, np.zeros(self.nb_actions)])
        index_control = len(self.agent.zeta) + action
        zeta_u[index_control] = 1

        # f depends on zeta and u.
        zeta_u_to_f = torch.from_numpy(zeta_u).float()
        zeta_to_J = torch.from_numpy(self.agent.zeta).float()

        # Those lines are only used to accelerate the computations but are not
        # strickly necessary.
        # because we don't want to compute the gradient wrt theta_f and theta_J.
        for param in self.net_f.parameters():
            param.requires_grad = False
        for param in self.net_J.parameters():
            param.requires_grad = False

        # In the Hamilton Jacobi Bellman equation, we derivate J by zeta.
        # But we do not want to propagate this gradient.
        # We seek to compute the gradient of J with respect to zeta_to_J.
        zeta_to_J.requires_grad = True
        # zeta_u_to_f.require_grad = False : This is already the default.

        # Deactivate dropout and batchnorm but continues to accumulate the gradient.
        # This is the reason it is generally used paired with "with torch.no_grad()"
        self.net_J.eval()
        self.net_f.eval()

        # in the no_grad context, all the results of the computations will have
        # requires_grad=False,
        # even if the inputs have requires_grad=True
        # If you want to freeze part of your model and train the rest, you can set
        # requires_grad of the parameters you want to freeze to False.
        f = self.net_f.forward(zeta_u_to_f).detach().numpy()
        new_zeta_ = self.agent.zeta + self.time_step * f
        instant_reward = self.agent.drive(new_zeta_)
        grad_ = torch.autograd.grad(
            self.net_J(zeta_to_J), zeta_to_J)[0]
        future_reward = torch.dot(grad_, self.net_f.forward(zeta_u_to_f))
        future_reward = future_reward.detach().numpy()

        score = instant_reward + future_reward

        zeta_to_J.requires_grad = False

        for param in self.net_f.parameters():
            param.requires_grad = True
        for param in self.net_J.parameters():
            param.requires_grad = True

        self.net_J.train()
        self.net_f.train()
        return score

    def simulation_one_step(self, k):
        """Simulate one step.

        Paramaters:
        -----------
        k: int

        Returns
        -------
        (action, loss): int, np.ndarray"""
        # if you are exacly on 0 (empty resource) you get stuck
        # because of the nature of the differential equation.
        for i in range(6):
            # zeta = x - x_star
            if self.agent.zeta[i] + self.agent.x_star[i] < self.min_resource:
                self.agent.zeta[i] = -self.agent.x_star[i] + self.min_resource

        possible_actions = self.actions_possible()
        indexes_possible_actions = [i for i in range(
            self.nb_actions) if possible_actions[i]]

        # The default action is doing nothing. Like people in real life.
        action = 9

        if np.random.random() <= self.eps:
            action = np.random.choice(indexes_possible_actions)

        else:
            best_score = np.Inf
            for act in indexes_possible_actions:
                score = self.evaluate_action(act)
                if score < best_score:
                    best_score = score
                    action = act

        zeta_to_nn = torch.from_numpy(self.agent.zeta).float()
        zeta_u = np.concatenate(
            [self.agent.zeta, np.zeros(self.nb_actions)])
        zeta_u[len(self.agent.zeta) + action] = 1
        zeta_u_to_nn = torch.from_numpy(zeta_u).float()

        new_zeta = self.new_state(action)  # actual choosen new_zeta
        new_zeta_to_nn = torch.from_numpy(new_zeta).float()

        coeff = self.asym_coeff
        # set of big actions leading directly to the resources and 4 is for sleeping
        if action in {4, 10, 11, 12, 13}:
            coeff = 1

        predicted_new_zeta = zeta_to_nn + self.time_step * \
            self.net_f.forward(zeta_u_to_nn)

        Loss_f = coeff * torch.dot(new_zeta_to_nn - predicted_new_zeta,
                                   new_zeta_to_nn - predicted_new_zeta)

        self.optimizer_J.zero_grad()
        self.optimizer_f.zero_grad()
        Loss_f.backward()
        self.optimizer_J.zero_grad()
        self.optimizer_f.step()

        zeta_to_nn.requires_grad = True

        # if drive = d(\zeta_t)= 1 and globally convex environment (instant
        # and long-term improvements are in the same direction)

        # futur drive = d(\zeta_t, u_a) = 0.9
        instant_drive = self.agent.drive(new_zeta)

        # negative
        delta_deviation = torch.dot(torch.autograd.grad(self.net_J(zeta_to_nn),
                                                        zeta_to_nn)[0],
                                    self.net_f.forward(zeta_u_to_nn))

        # 0.1 current deviation
        discounted_deviation = - np.log(self.gamma) * \
            self.net_J.forward(zeta_to_nn)
        Loss_J = torch.square(
            instant_drive + delta_deviation - discounted_deviation)

        zeta_to_nn.requires_grad = False

        self.optimizer_J.zero_grad()
        self.optimizer_f.zero_grad()
        Loss_J.backward()
        self.optimizer_f.zero_grad()
        self.optimizer_J.step()

        self.agent.zeta = new_zeta

        if (k % self.N_print) == 0:
            print("Iteration:", k, "/", self.N_iter - 1)
            print("Action:", action)
            print("zeta:", self.agent.zeta)
            print("")

        if (k % self.N_save_weights) == 0:
            torch.save(self.net_J.state_dict(), 'weights_net_J')
            torch.save(self.net_f.state_dict(), 'weights_net_f')

        loss = np.array([Loss_f.detach().numpy(), Loss_J.detach().numpy()[0]])
        return action, loss

    def compute_mask(self, scale):
        """Compute the mask.

        Parameters:
        -----------
        scale: int

        Returns:
        --------
        is_inside: np.ndarray
        """
        n_X = 9*scale
        n_Y = 6*scale
        values = np.empty((n_X, n_Y))
        values.fill(np.nan)
        is_inside = np.zeros((n_X, n_Y))
        for i in range(n_X):  # x
            for j in range(n_Y):  # y
                is_inside[i, j] = env.is_point_inside(i/scale, j/scale)
        return is_inside

    def plot_J(self, ax, fig, resource, scale, is_inside):
        """Plot of the learned J function.

        Parameters:
        -----------
        ax: SubplotBase
        resource: int
            1, 2, 3, or 4.
        scale: int
        is_inside : np.ndarray

        Returns:
        --------
        None
        """

        algo.net_J.eval()

        n_X = 9*scale
        n_Y = 6*scale
        values = np.empty((n_X, n_Y))
        values.fill(np.nan)
        for i in range(n_X):  # x
            for j in range(n_Y):  # y
                if is_inside[i, j]:
                    # We are at the optimum for three out of the 4 resources
                    # but one resources varies alongside with the coordinates.
                    # No muscular nor energic fatigues.
                    zeta = np.array(
                        [0, 0, 0, -self.agent.x_star[3], 0, 0, i/scale, j/scale, 0])
                    zeta = np.array([0, 0, 0, 0, 0, 0, i/scale, j/scale, 0])
                    zeta[resource-1] = -self.agent.x_star[resource-1]
                    zeta_to_J = torch.from_numpy(zeta).float()
                    values[i, j] = algo.net_J(zeta_to_J).detach().numpy()

        im = ax.imshow(X=values.T, cmap="YlGnBu", norm=Normalize())
        ax.axis('off')
        ax.invert_yaxis()
        ax.set_title(f'Deviation function (resource {resource} missing)')
        cbar = fig.colorbar(im, extend='both', shrink=0.4, ax=ax)

    def plot_ressources(self, ax, frame):
        """Plot the historic of the ressrouce with time in abscisse.

        Parameters:
        -----------
        ax: SubplotBase
        frame: int

        Returns:
        --------
        None
        Warning : abscisse is not time but step!
        """

        zeta_meaning = [
            "resource_1",
            "resource_2",
            "resource_3",
            "resource_4",
            "muscular energy",
            "aware energy",
            "x",
            "y",
            "angle",
        ]

        df = pd.DataFrame(self.historic_zeta[:frame+1],
                          columns=zeta_meaning)
        df.plot(ax=ax, grid=True, yticks=list(range(0, 10)))  # TODO
        ax.set_ylabel('value')
        ax.set_xlabel('frames')
        ax.set_title("Evolution of the resource")

    def plot_loss(self, ax, frame):
        """Plot the loss in order to control the learning of the agent.

        Parameters:
        -----------
        ax: SubplotBase
        frame: int

        Returns:
        --------
        None
        Warning : abscisse is not time but step!
        """

        loss_meaning = [
            "Loss of the transition function $L_f$",
            "Loss of the deviation function $L_J$",
        ]

        df = pd.DataFrame(self.historic_losses[:frame+1],
                          columns=loss_meaning)
        df = df.rolling(window=self.N_rolling).mean()
        df.plot(ax=ax, grid=True, logy=True)
        ax.set_ylabel('value of the losses')
        ax.set_xlabel('frames')
        ax.set_title(
            f"Evolution of the log-loss (moving average with {N_rolling} frames)")

    def plot_position(self, ax, zeta, controls_turn):
        """Plot the position with an arrow.

        Parameters:
        -----------
        ax: SubplotBase
        zeta: np.ndarray
        controls_turn: ?

        Returns:
        --------
        None
        Warning : abscisse is not time but step!
        """
        self.env.plot(ax=ax)  # initialisation of plt with background
        x = zeta[6]
        y = zeta[7]

        num_angle = int(zeta[8])

        dx, dy = controls_turn[num_angle]

        alpha = 0.5

        ax.arrow(x, y, dx, dy, head_width=0.1, alpha=alpha)
        ax.set_title("Position and orientation of the agent")

    def plot(self, frame,  scale=5):
        """Plot the position, angle and the ressources of the agent.

        - time, ressources, historic in transparence -> faire une fonction plot en dehors de l'agent


        Parameters:
        -----------
        frame :int

        Returns:
        --------
        None
        """

        is_inside = self.compute_mask(scale=scale)

        zeta = self.historic_zeta[frame]

        fig = plt.figure(figsize=(16, 16))
        shape = (4, 4)
        ax_resource = plt.subplot2grid(shape, (0, 0), colspan=4)
        ax_env = plt.subplot2grid(shape, (1, 0), colspan=2, rowspan=2)
        ax_loss = plt.subplot2grid(shape, (1, 2), colspan=2, rowspan=2)
        axs_J = [None]*4
        axs_J[0] = plt.subplot2grid(shape, (3, 0))
        axs_J[1] = plt.subplot2grid(shape, (3, 1))
        axs_J[2] = plt.subplot2grid(shape, (3, 2))
        axs_J[3] = plt.subplot2grid(shape, (3, 3))

        last_action = self.historic_actions[frame]

        fig.suptitle(
            (f'Dashboard. Frame: {frame} - last action: '
                f'{last_action}: {meaning_actions[last_action]} '),
            fontsize=16)

        self.plot_position(ax=ax_env, zeta=zeta,
                           controls_turn=controls_turn)

        self.plot_ressources(ax=ax_resource, frame=frame)
        self.plot_loss(ax=ax_loss, frame=frame)

        for resource in range(4):
            self.plot_J(ax=axs_J[resource],
                        fig=fig, resource=resource+1,
                        scale=scale, is_inside=is_inside)

        plt.tight_layout()
        name_fig = f"images/frame_{frame}"
        plt.savefig(name_fig)
        print(name_fig)
        plt.close(fig)

    def simulation(self):

        for k in range(self.N_iter):
            print(k)
            action, loss = self.simulation_one_step(k)

            # save historic
            self.historic_zeta.append(self.agent.zeta)
            self.historic_actions.append(action)
            self.historic_losses.append(loss)

            if k % self.cycle_plot == 0:
                self.plot(k)

        torch.save(self.net_J.state_dict(), 'weights_net_J')
        torch.save(self.net_f.state_dict(), 'weights_net_f')


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
