import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import Normalize
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns

class Environment:
    def __init__(self, coord_env, coord_circ):
        self.coord_env = coord_env
        self.coord_circ = coord_circ

    def is_point_inside(self, x, y):
        """Check if a point (x,y) is inside the polygon"""
        lab = list(self.coord_env.keys())
        n_left = 0
        for i in range(0, len(lab) - 2, 2):
            if (((self.coord_env[lab[i + 1]] > y) and (self.coord_env[lab[i + 3]] <= y)) or ((self.coord_env[lab[i + 1]] <= y) and (self.coord_env[lab[i + 3]] > y))) and (self.coord_env[lab[i]] <= x):
                n_left += 1
        if (((self.coord_env[lab[len(lab) - 1]] > y) and (self.coord_env[lab[1]] <= y)) or ((self.coord_env[lab[len(lab) - 1]] <= y) and (self.coord_env[lab[1]] > y))) and (self.coord_env[lab[len(lab) - 1]] <= x):
            n_left += 1
        if n_left % 2 == 1:
            return True
        else:
            return False

    def is_segment_inside(self, xa, xb, ya, yb):
        """Check if the segment AB with A(xa, ya) and B(xb, yb) is completely inside
        the polygon"""
        lab = list(self.coord_env.keys()) + list(self.coord_env.keys())[:2]
        n_inter = 0
        if (xa != xb):
            alpha_1 = (yb - ya) / (xb - xa)
            beta_1 = (ya * xb - yb * xa) / (xb - xa)
            for i in range(0, len(lab) - 2, 2):
                if self.coord_env[lab[i]] == self.coord_env[lab[i + 2]]:
                    inter = [self.coord_env[lab[i]], alpha_1 *
                             self.coord_env[lab[i]] + beta_1]
                    inter_in_AB = (inter[0] >= np.minimum(xa, xb)) and (inter[0] <= np.maximum(
                        xa, xb)) and (inter[1] >= np.minimum(ya, yb)) and (inter[1] <= np.maximum(ya, yb))
                    inter_in_border = (inter[0] >= np.minimum(self.coord_env[lab[i]], self.coord_env[lab[i + 2]])) and (inter[0] <= np.maximum(self.coord_env[lab[i]], self.coord_env[lab[i + 2]])) and (
                        inter[1] >= np.minimum(self.coord_env[lab[i + 1]], self.coord_env[lab[i + 3]])) and (inter[1] <= np.maximum(self.coord_env[lab[i + 1]], self.coord_env[lab[i + 3]]))
                    if inter_in_AB and inter_in_border:
                        n_inter += 1
                else:
                    if ya == yb:
                        if ya == self.coord_env[lab[i + 1]]:
                            if (np.minimum(xa, xb) <= np.maximum(self.coord_env[lab[i]], self.coord_env[lab[i + 2]])) and (np.maximum(xa, xb) >= np.minimum(self.coord_env[lab[i]], self.coord_env[lab[i + 2]])):
                                n_inter += 1
                    else:
                        inter = [(self.coord_env[lab[i + 1]] - beta_1) /
                                 alpha_1, self.coord_env[lab[i + 1]]]
                        inter_in_AB = (inter[0] >= np.minimum(xa, xb)) and (inter[0] <= np.maximum(
                            xa, xb)) and (inter[1] >= np.minimum(ya, yb)) and (inter[1] <= np.maximum(ya, yb))
                        inter_in_border = (inter[0] >= np.minimum(self.coord_env[lab[i]], self.coord_env[lab[i + 2]])) and (inter[0] <= np.maximum(self.coord_env[lab[i]], self.coord_env[lab[i + 2]])) and (
                            inter[1] >= np.minimum(self.coord_env[lab[i + 1]], self.coord_env[lab[i + 3]])) and (inter[1] <= np.maximum(self.coord_env[lab[i + 1]], self.coord_env[lab[i + 3]]))
                        if inter_in_AB and inter_in_border:
                            n_inter += 1
        else:
            for i in range(0, len(lab) - 2, 2):
                if self.coord_env[lab[i]] == self.coord_env[lab[i + 2]]:
                    if xa == self.coord_env[lab[i]]:
                        if (np.minimum(ya, yb) <= np.maximum(self.coord_env[lab[i + 1]], self.coord_env[lab[i + 3]])) and (np.maximum(ya, yb) >= np.minimum(self.coord_env[lab[i + 1]], self.coord_env[lab[i + 3]])):
                            n_inter += 1
                else:
                    inter = [xa, self.coord_env[lab[i + 1]]]
                    inter_in_AB = (inter[0] >= np.minimum(xa, xb)) and (inter[0] <= np.maximum(
                        xa, xb)) and (inter[1] >= np.minimum(ya, yb)) and (inter[1] <= np.maximum(ya, yb))
                    inter_in_border = (inter[0] >= np.minimum(self.coord_env[lab[i]], self.coord_env[lab[i + 2]])) and (inter[0] <= np.maximum(self.coord_env[lab[i]], self.coord_env[lab[i + 2]])) and (
                        inter[1] >= np.minimum(self.coord_env[lab[i + 1]], self.coord_env[lab[i + 3]])) and (inter[1] <= np.maximum(self.coord_env[lab[i + 1]], self.coord_env[lab[i + 3]]))
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

        if ax is None:
            ax = plt.subplot(111)

        lab = list(self.coord_env.keys())
        for i in range(0, len(lab) - 2, 2):
            ax.plot([self.coord_env[lab[i]], self.coord_env[lab[i + 2]]],
                    [self.coord_env[lab[i + 1]], self.coord_env[lab[i + 3]]],
                    '-', color='black', lw=2)
        ax.plot([self.coord_env[lab[len(lab) - 2]], self.coord_env[lab[0]]],
                [self.coord_env[lab[len(lab) - 1]], self.coord_env[lab[1]]],
                '-', color='black', lw=2)

        for circle_name, circle in self.coord_circ.items():
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
                 time_step, eps, gamma, tau, N_iter, N_save_weights, N_print, learning_rate,
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
        action: int"""
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
            print("Iteration:", k, "/", self.N_iter)
            print("Action:", action)
            print("zeta:", self.agent.zeta)
            print("")

        if (k % self.N_save_weights) == 0:
            torch.save(self.net_J.state_dict(), 'weights_net_J')
            torch.save(self.net_f.state_dict(), 'weights_net_f')

        return action

    def plot_J(self, ax, fig, resource, scale=5):
        """Plot of the learned J function.

        Parameters:
        -----------
        ax: SubplotBase
        resource: int
            1, 2, 3, or 4.
        scale: int

        Returns:
        --------
        None
        """

        algo.net_J.eval()

        n_X = 9*scale
        n_Y = 6*scale
        values = np.empty((n_X, n_Y))
        values.fill(np.nan)
        mask = np.zeros((n_X, n_Y))
        for i in range(n_X):  # x
            for j in range(n_Y):  # y
                inside = env.is_point_inside(i/scale, j/scale) # TODO : factoriser
                if not inside:
                    mask[i, j] = True
                else:
                    # We are at the optimum for three out of the 4 resources
                    # but one resources is at its lowest.
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
        ax.set_title(f'Resource {resource}')
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

        import pandas as pd

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

    def plot(self, n_step: int = 1):
        """Plot the position, angle and the ressources of the agent.

        - time, ressources, historic in transparence -> faire une fonction plot en dehors de l'agent


        Parameters:
        -----------
        n_step: We save the figure each n_step.

        Returns:
        --------
        None
        """

        for frame, zeta in enumerate(self.historic_zeta[:-1:n_step]):

            # fig, axs = plt.subplots(1, 3, figsize=(15, 9), sharey=True,
            #                         gridspec_kw={'width_ratios': [3, 1, 3]})

            fig = plt.figure(figsize=(15, 9))
            ax_env = plt.subplot2grid((3, 3), (0, 0), colspan=3)
            ax_resource = plt.subplot2grid((3, 3), (1, 0), colspan=2)
            ax_J = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
            ax4 = plt.subplot2grid((3, 3), (2, 0))
            ax5 = plt.subplot2grid((3, 3), (2, 1))

            last_action = self.historic_actions[frame]

            fig.suptitle(
                f'frame {frame}- last action: {last_action} : {meaning_actions[last_action]} ',
                fontsize=16)

            # ax_env = axs[0]
            # ax_resource = axs[1]
            # ax_J = axs[2]

            self.env.plot(ax=ax_env)  # initialisation of plt with background

            self.plot_ressources(ax=ax_resource, frame=frame)
            self.plot_J(ax=ax_J, fig=fig, resource=1)  # todo: boucle.

            x = zeta[6]
            y = zeta[7]

            num_angle = int(zeta[8])

            dx, dy = controls_turn[num_angle]

            alpha = 0.5

            ax_env.arrow(x, y, dx, dy, head_width=0.1, alpha=alpha)
            # todo: add circle
            name_fig = f"images/frame_{frame}"
            plt.savefig(name_fig)
            print(name_fig)
            plt.close(fig)

    # def animate(self):
        # from matplotlib import animation

        # plt_background = self.env.plot()

        # # First set up the figure, the axis, and the plot element we want to animate
        # fig = plt_background.figure()
        # plt = plt_background
        # ax = plt.axes(xlim=(-1, 10), ylim=(-1, 10))
        # line, = ax.plot([], [], lw=2)

        # # initialization function: plot the background of each frame
        # def init():
        #     line.set_data([], [])
        #     return line,

        # # animation function.  This is called sequentially
        # def animate(i):
        #     # zeta = self.historic_zeta[i]
        #     # x = zeta[6], zeta[6]
        #     # y = zeta[7], zeta[7]

        #     x = np.array(self.historic_zeta)[:i, 6]
        #     y = np.array(self.historic_zeta)[:i, 7]
        #     line.set_data(x, y)
        #     line.set_data(x, y)

        #     return line,

        # # call the animator.  blit=True means only re-draw the parts that have changed.
        # anim = animation.FuncAnimation(fig, animate, init_func=init,
        #                                frames=len(self.historic_zeta)-1,
        #                                interval=20, blit=True)

        # # save the animation as an mp4.  This requires ffmpeg or mencoder to be
        # # installed.  The extra_args ensure that the x264 codec is used, so that
        # # the video can be embedded in html5.  You may need to adjust this for
        # # your system: for more information, see
        # # http://matplotlib.sourceforge.net/api/animation_api.html
        # anim.save('basic_animation.mp4', fps=30)

        # plt.show()

        # from matplotlib.animation import FuncAnimation

        # fig, ax = plt.subplots()
        # xdata, ydata = [], []
        # ln, = plt.plot([], [], 'ro')

        # def init():
        #     ax.set_xlim(0, 2*np.pi)
        #     ax.set_ylim(-1, 1)
        #     ax = self.env.plot(ax)
        #     return ln,

        # def update(frame):
        #     xdata.append(frame)
        #     ydata.append(np.sin(frame))
        #     ln.set_data(xdata, ydata)
        #     return ln,

        # ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 128),
        #                     init_func=init, blit=True)
        # plt.show()

    def simulation(self):

        for k in range(self.N_iter):
            print(k)
            action = self.simulation_one_step(k)

            # save historic
            self.historic_zeta.append(self.agent.zeta)
            self.historic_actions.append(action)

        self.plot()
        # plt.show()

        # self.animate()
        # plt.show()

        torch.save(self.net_J.state_dict(), 'weights_net_J')
        torch.save(self.net_f.state_dict(), 'weights_net_f')


# Simulation
coord_env = {'xa': 1, 'ya': 1,
             'xb': 1, 'yb': 5,
             'xc': 2, 'yc': 5,
             'xd': 2, 'yd': 3,
             'xe': 3, 'ye': 3,
             'xf': 3, 'yf': 4,
             'xg': 4, 'yg': 4,
             'xh': 4, 'yh': 6,
             'xi': 9, 'yi': 6,
             'xj': 9, 'yj': 5,
             'xk': 6, 'yk': 5,
             'xl': 6, 'yl': 3,
             'xm': 7, 'ym': 3,
             'xn': 7, 'yn': 0,
             'xo': 6, 'yo': 0,
             'xp': 6, 'yp': 2,
             'xq': 5, 'yq': 2,
             'xr': 5, 'yr': 1}

# x, y, r, color
coord_circ = {'circle_1': [1.5, 4.25, 0.3, 'red'],
              'circle_2': [4.5, 1.5, 0.3, 'blue'],
              'circle_3': [8, 5.5, 0.3, 'orange'],
              'circle_4': [6.5, 0.75, 0.3, 'green']}

env = Environment(coord_env, coord_circ)

# homeostatic point
# Resources 1, 2, 3, 4 and muscular fatigues and sleep fatigue
x_star = np.array([1, 2, 3, 4, 0, 0])

# parameters of the function f
# same + x, y, and angle coordinates
c = np.array([-0.05, -0.05, -0.05, -0.05, -0.008, 0.0005, 0, 0, 0])

# Not used currently
angle_visual_field = np.pi / 10

agent = Agent(x_star, c, angle_visual_field)


n_neurons = 128
dropout_rate = 0.15

net_J = Net_J(n_neurons, dropout_rate)
net_f = Net_f(n_neurons, dropout_rate)


time_step = 1
eps = 0.3
gamma = 0.99
tau = 0.001  # not used yet (linked with the target function)
N_iter = 3
N_save_weights = 1000  # save neural networks weights every N_save_weights step
N_print = 1
learning_rate = 0.001


# We discretized the angles in order no to spin without moving
# The controls for walking and running for each angle is pre-computed
num_pos_angles = 5
controls_turn = [[np.cos(2 * np.pi / num_pos_angles * i),
                  np.sin(2 * np.pi / num_pos_angles * i)]
                 for i in range(num_pos_angles)]

control_walking = [np.array([0, 0, 0, 0, 0.01, 0,  # homeostatic resources
                             0.1 * controls_turn[i][0],  # x
                             0.1 * controls_turn[i][1],  # y
                             0])  # angle
                   for i in range(num_pos_angles)]
control_running = [np.array([0, 0, 0, 0, 0.05, 0,
                             0.3 * controls_turn[i][0],
                             0.3 * controls_turn[i][1],
                             0])
                   for i in range(num_pos_angles)]


# For each action, there is a control verifying for example that
# the muscular tiredness enables the agent to walk.
# for example, for the first action (walking), we verify that the tiredness is not above 6.
# For the second action (running), we verify that the tiredness is not above 3.
actions_controls = [
    # walking
    control_walking,  # -> constraints[0]
    # running
    control_running,  # -> constraints[1]
    # turning an angle to the left
    np.array([0, 0, 0, 0, 0.001, 0, 0, 0, 0]),  # etc...
    # turning an angle to the right
    np.array([0, 0, 0, 0, 0.001, 0, 0, 0, 0]),
    # sleeping
    np.array([0, 0, 0, 0, 0, -0.001, 0, 0, 0]),
    # get resource 1
    np.array([0.1, 0, 0, 0, 0, 0, 0, 0, 0]),
    # get resource 2
    np.array([0, 0.1, 0, 0, 0, 0, 0, 0, 0]),
    # get resource 3
    np.array([0, 0, 0.1, 0, 0, 0, 0, 0, 0]),
    # get resource 4
    np.array([0, 0, 0, 0.1, 0, 0, 0, 0, 0]),
    # not doing anything
    np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])]
# Keep in mind that the agent looses resources and energy even
# if he does nothing via the function f.

# there are 4 additionnal actions : Going to the 4 resource and eating
nb_actions = len(actions_controls) + 4


meaning_actions = {
    0: "walking",
    1: "running",
    2: "turning trigo",
    3: "turning anti trigo",
    4: "sleeping",
    5: "get resource 1",
    6: "get resource 2",
    7: "get resource 3",
    8: "get resource 4",
    9: "not doing anything",
    10: "going direcly to resource 1",
    11: "going direcly to resource 2",
    12: "going direcly to resource 3",
    13: "going direcly to resource 4",
}


# Same order of action as actions_controls
# Constraint verify the tiredness. There are tree types of tiredness:
# - muscular tiredness (M)
# - and sleep tiredness (S)
# - max_food_in_the_stomach (F)
#                M, M, M, M, S, F,  F,  F,  F,  (*)
# constraints = [6, 3, 6, 6, 1, 15, 15, 15, 15, None]
constraints = [6, 3, 6, 6, 1, 8, 8, 8, 8, None]
# (*) There is not any constraint when you do anything.

min_time_sleep = 1000  # ratio of the minimum sleep time for the agent and the time_step
max_tired = 10
asym_coeff = 100
min_resource = 0.1

algo = Algorithm(env, agent, net_J, net_f,
                 time_step, eps, gamma, tau, N_iter, N_save_weights, N_print, learning_rate,
                 actions_controls, constraints, min_time_sleep, max_tired, asym_coeff,
                 min_resource, nb_actions)


algo.simulation()
