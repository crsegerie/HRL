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
