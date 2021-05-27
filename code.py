import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def plot(self, save_fig=False):
        lab = list(self.coord_env.keys())
        for i in range(0, len(lab) - 2, 2):
            plt.plot([self.coord_env[lab[i]], self.coord_env[lab[i + 2]]],
                     [self.coord_env[lab[i + 1]], self.coord_env[lab[i + 3]]], '-', color='black', lw=2)
        plt.plot([self.coord_env[lab[len(lab) - 2]], self.coord_env[lab[0]]],
                 [self.coord_env[lab[len(lab) - 1]], self.coord_env[lab[1]]], '-', color='black', lw=2)

        for key in self.coord_circ:
            circle = plt.Circle((self.coord_circ[key][0], self.coord_circ[key][1]),
                                self.coord_circ[key][2], color=self.coord_circ[key][3])
            plt.gcf().gca().add_artist(circle)

        plt.axis('off')

        if save_fig:
            plt.savefig('environment.eps', format='eps')


class Agent:
    def __init__(self, x_star, a, angle_visual_field):
        self.x_star = x_star
        self.zeta = np.zeros(9)
        self.zeta[6] = 3  # initialization position for the agent
        self.zeta[7] = 2  # initialization position for the agent
        self.a = a
        self.angle_visual_field = angle_visual_field

    def dynamics(self, zeta, u):
        f = np.zeros(zeta.shape)
        f[:6] = self.a[:6] * (zeta[:6] + self.x_star) + \
            u[:6] * (zeta[:6] + self.x_star)
        f[6:9] = u[6:9]
        return f

    def drive(self, zeta, epsilon=0.001):
        delta = zeta[:6]
        drive_delta = np.sqrt(epsilon + np.dot(delta, delta))
        return drive_delta


class Net_J(nn.Module):
    """"""

    def __init__(self, n_neurons, dropout_rate):
        super(Net_J, self).__init__()
        self.fc1 = nn.Linear(9, n_neurons)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(n_neurons, n_neurons)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(n_neurons, 1)

    def forward(self, x):
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
    def __init__(self, n_neurons, dropout_rate):
        super(Net_f, self).__init__()
        self.fc1 = nn.Linear(9 + 14, n_neurons)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(n_neurons, n_neurons)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(n_neurons, 9)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout1(x)
        output = self.fc3(x)
        return output


# Algorithm

class Algorithm:
    def __init__(self, env, agent, net_J, net_f,
                 time_step, eps, gamma, tau, N_iter, N_save_weights, N_print, learning_rate,
                 actions_controls, constraints, min_time_sleep, max_tired):

        self.env = env
        self.agent = agent
        self.net_J = net_J
        self.net_f = net_f

        self.time_step = time_step
        self.eps = eps
        self.gamma = gamma
        self.tau = tau
        self.N_iter = N_iter
        self.N_save_weights = N_save_weights
        self.N_print = N_print
        self.learning_rate = learning_rate
        self.optimizer_J = torch.optim.Adam(
            self.net_J.parameters(), lr=self.learning_rate)
        self.optimizer_f = torch.optim.Adam(
            self.net_f.parameters(), lr=self.learning_rate)

        self.actions_controls = actions_controls
        self.constraints = constraints
        self.min_time_sleep = min_time_sleep
        self.max_tired = max_tired

    def actions_possible(self):
        # + 4 for the action of going to a resource after seeing it
        possible_actions = [True for i in range(
            len(self.actions_controls) + 4)]

        if self.agent.zeta[4] >= self.constraints[0]:
            possible_actions[0] = False
        if not self.env.is_point_inside(self.agent.zeta[6] + self.time_step * self.actions_controls[0][6] * np.cos(self.agent.zeta[8]), self.agent.zeta[7] + self.time_step * self.actions_controls[0][7] * np.sin(self.agent.zeta[8])):
            possible_actions[0] = False

        if self.agent.zeta[4] >= self.constraints[1]:
            possible_actions[1] = False
        if not self.env.is_point_inside(self.agent.zeta[6] + self.time_step * self.actions_controls[1][6] * np.cos(self.agent.zeta[8]), self.agent.zeta[7] + self.time_step * self.actions_controls[1][7] * np.sin(self.agent.zeta[8])):
            possible_actions[1] = False

        if self.agent.zeta[4] >= self.constraints[2]:
            possible_actions[2] = False

        if self.agent.zeta[4] >= self.constraints[3]:
            possible_actions[3] = False

        if self.agent.zeta[5] <= self.constraints[4]:
            possible_actions[4] = False
        if self.agent.zeta[5] >= self.max_tired:
            possible_actions = [False for i in range(
                len(self.actions_controls) + 4)]
            possible_actions[4] = True

        if self.agent.zeta[0] >= self.constraints[5]:
            possible_actions[5] = False
        if (self.agent.zeta[6] - self.env.coord_circ['circle_1'][0])**2 + (self.agent.zeta[7] - self.env.coord_circ['circle_1'][1])**2 > self.env.coord_circ['circle_1'][2]**2:
            possible_actions[5] = False

        if self.agent.zeta[1] >= self.constraints[6]:
            possible_actions[6] = False
        if (self.agent.zeta[6] - self.env.coord_circ['circle_2'][0])**2 + (self.agent.zeta[7] - self.env.coord_circ['circle_2'][1])**2 > self.env.coord_circ['circle_2'][2]**2:
            possible_actions[6] = False

        if self.agent.zeta[2] >= self.constraints[7]:
            possible_actions[7] = False
        if (self.agent.zeta[6] - self.env.coord_circ['circle_3'][0])**2 + (self.agent.zeta[7] - self.env.coord_circ['circle_3'][1])**2 > self.env.coord_circ['circle_3'][2]**2:
            possible_actions[7] = False

        if self.agent.zeta[3] >= self.constraints[8]:
            possible_actions[8] = False
        if (self.agent.zeta[6] - self.env.coord_circ['circle_4'][0])**2 + (self.agent.zeta[7] - self.env.coord_circ['circle_4'][1])**2 > self.env.coord_circ['circle_4'][2]**2:
            possible_actions[8] = False

        if not self.env.is_segment_inside(self.agent.zeta[6], self.env.coord_circ['circle_1'][0], self.agent.zeta[7], self.env.coord_circ['circle_1'][1]):
            possible_actions[10] = False

        if not self.env.is_segment_inside(self.agent.zeta[6], self.env.coord_circ['circle_2'][0], self.agent.zeta[7], self.env.coord_circ['circle_2'][1]):
            possible_actions[11] = False

        if not self.env.is_segment_inside(self.agent.zeta[6], self.env.coord_circ['circle_3'][0], self.agent.zeta[7], self.env.coord_circ['circle_3'][1]):
            possible_actions[12] = False

        if not self.env.is_segment_inside(self.agent.zeta[6], self.env.coord_circ['circle_4'][0], self.agent.zeta[7], self.env.coord_circ['circle_4'][1]):
            possible_actions[13] = False

        return possible_actions

    def new_state(self, a):
        if a == 4:
            new_zeta = self.agent.zeta
            return new_zeta

        else:
            u = np.zeros(self.agent.zeta.shape)

            if a == 0:
                u = self.actions_controls[0]
                u[6] = u[6] * np.cos(self.agent.zeta[8])
                u[7] = u[7] * np.sin(self.agent.zeta[8])

            elif a == 1:
                u = self.actions_controls[1]
                u[6] = u[6] * np.cos(self.agent.zeta[8])
                u[7] = u[7] * np.sin(self.agent.zeta[8])

            elif (a == 2) or (a == 3) or (a == 5) or (a == 6) or (a == 7) or (a == 8) or (a == 9):
                u = self.actions_controls[a]

            new_zeta = self.agent.zeta + self.time_step * \
                self.agent.dynamics(self.agent.zeta, u)
            new_zeta[8] = new_zeta[8] % (2 * np.pi)

            return new_zeta

    def simulation(self):
        for k in range(N_iter):
            possible_actions = self.actions_possible()
            indexes_possible_actions = [i for i in range(
                len(possible_actions)) if possible_actions[i]]
            action = 9

            if np.random.random() <= self.eps:
                action = np.random.choice(indexes_possible_actions)

            else:
                best_score = np.Inf
                for i in range(len(indexes_possible_actions)):
                    zeta_u = np.concatenate(
                        [self.agent.zeta, np.zeros(len(possible_actions))])
                    zeta_u[len(self.agent.zeta) +
                           indexes_possible_actions[i]] = 1
                    zeta_u_to_f = torch.from_numpy(zeta_u).float()
                    zeta_to_J = torch.from_numpy(self.agent.zeta).float()
                    zeta_to_J.requires_grad = True

                    self.net_J.eval()
                    self.net_f.eval()

                    score = self.agent.drive(
                        self.agent.zeta + self.time_step * self.net_f.forward(zeta_u_to_f).detach().numpy())
                    score += torch.dot(torch.autograd.grad(self.net_J(zeta_to_J), zeta_to_J)[
                                       0], self.net_f.forward(zeta_u_to_f)).detach().numpy()

                    zeta_to_J.requires_grad = False

                    self.net_J.train()
                    self.net_f.train()

                    if score < best_score:
                        best_score = score
                        action = indexes_possible_actions[i]

            zeta_to_nn = torch.from_numpy(self.agent.zeta).float()
            zeta_u = np.concatenate(
                [self.agent.zeta, np.zeros(len(possible_actions))])
            zeta_u[len(self.agent.zeta) + action] = 1
            zeta_u_to_nn = torch.from_numpy(zeta_u).float()
            new_zeta = self.new_state(action)
            new_zeta_to_nn = torch.from_numpy(new_zeta).float()

            Loss_f = torch.dot(new_zeta_to_nn - zeta_to_nn - self.time_step * self.net_f.forward(
                zeta_u_to_nn), new_zeta_to_nn - zeta_to_nn - self.time_step * self.net_f.forward(zeta_u_to_nn))

            self.optimizer_J.zero_grad()
            self.optimizer_f.zero_grad()
            Loss_f.backward()
            self.optimizer_J.zero_grad()
            self.optimizer_f.step()

            zeta_to_nn.requires_grad = True
            Loss_J = torch.square(self.agent.drive(new_zeta) + torch.dot(torch.autograd.grad(self.net_J(zeta_to_nn), zeta_to_nn)[
                                  0], self.net_f.forward(zeta_u_to_nn)) + np.log(self.gamma) * self.net_J.forward(zeta_to_nn))
            zeta_to_nn.requires_grad = False

            self.optimizer_J.zero_grad()
            self.optimizer_f.zero_grad()
            Loss_J.backward()
            self.optimizer_f.zero_grad()
            self.optimizer_J.step()

            self.agent.zeta = new_zeta

            if k % self.N_print == 0:
                print("Iteration:", k, "/", self.N_iter)
                print("zeta:", self.agent.zeta)

            if k % self.N_save_weights == 0:
                torch.save(self.net_J.state_dict(), 'weights_net_J')
                torch.save(self.net_f.state_dict(), 'weights_net_f')

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

coord_circ = {'circle_1': [1.5, 4.25, 0.3, 'red'],
              'circle_2': [4.5, 1.5, 0.3, 'blue'],
              'circle_3': [8, 5.5, 0.3, 'orange'],
              'circle_4': [6.5, 0.75, 0.3, 'green']}

env = Environment(coord_env, coord_circ)


x_star = np.array([1, 2, 3, 4, 0, 0])
# parameters of the function f
a = np.array([0.05, 0.05, 0.05, 0.05, -0.008, 0.0005, 0, 0, 0])
angle_visual_field = np.pi / 10

agent = Agent(x_star, a, angle_visual_field)


n_neurons = 128
dropout_rate = 0.3

net_J = Net_J(n_neurons, dropout_rate)
net_f = Net_f(n_neurons, dropout_rate)


time_step = 1
eps = 0.1
gamma = 0.99
tau = 0.001
N_iter = 10000
N_save_weights = 1000  # save neural networks weights every N_save_weights step
N_print = 100
learning_rate = 0.001

actions_controls = [np.array([0, 0, 0, 0, 0.01, 0, 0.01, 0.01, 0]),  # walking
                    np.array([0, 0, 0, 0, 0.05, 0, 0.03, 0.03, 0]),  # running
                    # turning an angle to the left
                    np.array([0, 0, 0, 0, 0.001, 0, 0, 0, np.pi / 20]),
                    # turning an angle to the right
                    np.array([0, 0, 0, 0, 0.001, 0, 0, 0, -np.pi / 20]),
                    np.array([0, 0, 0, 0, 0, -0.001, 0, 0, 0]),  # sleeping
                    # consuming resource 1
                    np.array([0.1, 0, 0, 0, 0, 0, 0, 0, 0]),
                    # consuming resource 2
                    np.array([0, 0.1, 0, 0, 0, 0, 0, 0, 0]),
                    # consuming resource 3
                    np.array([0, 0, 0.1, 0, 0, 0, 0, 0, 0]),
                    # consuming resource 4
                    np.array([0, 0, 0, 0.1, 0, 0, 0, 0, 0]),
                    np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])]  # not doing anything

# same order of action as actions_controls
constraints = [6, 3, 6, 6, 1, 15, 15, 15, 15, None]

min_time_sleep = 1000  # ratio of the minimum sleep time for the agent and the time_step
max_tired = 10

algo = Algorithm(env, agent, net_J, net_f,
                 time_step, eps, gamma, tau, N_iter, N_save_weights, N_print, learning_rate,
                 actions_controls, constraints, min_time_sleep, max_tired)
