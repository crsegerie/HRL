"""Config file showing the meta parameters"""

import numpy as np


class Cfg_env:
    def __init__(self):

        # Simulation
        self.coord_env = {'xa': 1, 'ya': 1,
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
        self.coord_circ = {'circle_1': [1.5, 4.25, 0.3, 'red'],
                           'circle_2': [4.5, 1.5, 0.3, 'blue'],
                           'circle_3': [8, 5.5, 0.3, 'orange'],
                           'circle_4': [6.5, 0.75, 0.3, 'green']}


class Cfg_agent:
    def __init__(self):

        # homeostatic point
        # Resources 1, 2, 3, 4 and muscular fatigues and sleep fatigue
        self.x_star = np.array([1, 2, 3, 4, 0, 0])

        # parameters of the function f
        # same + x, y, and angle coordinates
        self.c = np.array(
            [-0.05, -0.05, -0.05, -0.05, -0.008, 0.0005, 0, 0, 0])

        # Not used currently
        self.angle_visual_field = np.pi / 10


class Cfg_nets:
    def __init__(self):

        self.n_neurons = 128
        self.dropout_rate = 0.15


class Cfg_algo:
    def __init__(self):

        # Iterations
        self.N_iter = 10
        self.time_step = 1

        # RL learning
        self.eps = 0.3  # random actions
        self.gamma = 0.99  # discounted rate
        self.tau = 0.001  # not used yet (linked with the target function)

        # Gradient descent
        self.learning_rate = 0.001
        self.asym_coeff = 100

        # plotting
        self.cycle_plot = self.N_iter - 1
        self.N_rolling = 5
        self.N_save_weights = 1000  # save neural networks weights every N_save_weights step
        self.N_print = 1


class Cfg_actions:
    def __init__(self):
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
        self.actions_controls = [
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
        self.nb_actions = len(self.actions_controls) + 4

        self.meaning_actions = {
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
        self.constraints = [6, 3, 6, 6, 1, 8, 8, 8, 8, None]
        # (*) There is not any constraint when you do anything.

        # ratio of the minimum sleep time for the agent and the time_step
        self.min_time_sleep = 1000
        self.max_tired = 10

        self.min_resource = 0.1
