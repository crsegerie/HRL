import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np


class Environment:
    def __init__(self):
        # Simulation
        self.coord_env = {
            'xa': 1, 'ya': 1,
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
        self.coord_circ = {
            'circle_1': [1.5, 4.25, 0.3, 'red'],
            'circle_2': [4.5, 1.5, 0.3, 'blue'],
            'circle_3': [8, 5.5, 0.3, 'orange'],
            'circle_4': [6.5, 0.75, 0.3, 'green']}

    def is_point_inside(self, x, y):
        """Check if a point (x,y) is inside the polygon.
        To do this, we look at the number of sides of the polygon at the left
        of the point.
        """

        coords = self.coord_env
        lab = list(coords.keys())
        n_left = 0

        def is_left(ind_x, ind_y_1, ind_y_2):
            cstr_1_y = (coords[lab[ind_y_1]] > y) and (
                coords[lab[ind_y_2]] <= y)
            cstr_2_y = (coords[lab[ind_y_1]] <= y) and (
                coords[lab[ind_y_2]] > y)
            cstr_x = (coords[lab[ind_x]] <= x)
            if (cstr_1_y or cstr_2_y) and cstr_x:
                return True
            return False

        for i in range(0, len(lab) - 2, 2):
            if is_left(i, i + 1, i + 3):
                n_left += 1
        if is_left(len(lab) - 2, len(lab) - 1, 1):
            n_left += 1
        if n_left % 2 == 1:
            return True
        else:
            return False

    def is_segment_inside(self, xa, xb, ya, yb):
        """Check if the segment AB with A(xa, ya) and B(xb, yb) is completely
        inside the polygon.
        To do this, we look at the number of intersections between the segment
        and the sides of the polygon.
        """

        coords = self.coord_env
        lab = list(coords.keys()) + list(coords.keys())[:2]
        n_inter = 0

        def is_inter(inter, ind):
            inter_in_AB = (inter[0] >= np.minimum(xa, xb)) and \
                          (inter[0] <= np.maximum(xa, xb)) and \
                          (inter[1] >= np.minimum(ya, yb)) and \
                          (inter[1] <= np.maximum(ya, yb))
            inter_in_border = (inter[0] >= np.minimum(coords[lab[ind]],
                                                      coords[lab[ind + 2]])) and \
                (inter[0] <= np.maximum(coords[lab[ind]],
                                        coords[lab[ind + 2]])) and \
                (inter[1] >= np.minimum(coords[lab[ind + 1]],
                                        coords[lab[ind + 3]])) and \
                (inter[1] <= np.maximum(coords[lab[ind + 1]],
                                        coords[lab[ind + 3]]))
            if inter_in_AB and inter_in_border:
                return True
            return False

        if (xa != xb):
            alpha_1 = (yb - ya) / (xb - xa)
            beta_1 = (ya * xb - yb * xa) / (xb - xa)
            for i in range(0, len(lab) - 2, 2):
                if coords[lab[i]] == coords[lab[i + 2]]:
                    inter = [coords[lab[i]], alpha_1 * coords[lab[i]] + beta_1]
                    if is_inter(inter, i):
                        n_inter += 1
                else:
                    if ya == yb:
                        if ya == coords[lab[i + 1]]:
                            inter_in_border = (np.minimum(xa, xb) <=
                                               np.maximum(coords[lab[i]],
                                                          coords[lab[i + 2]])) and \
                                              (np.maximum(xa, xb) >=
                                               np.minimum(coords[lab[i]],
                                                          coords[lab[i + 2]]))
                            if inter_in_border:
                                n_inter += 1
                    else:
                        inter = [(coords[lab[i + 1]] - beta_1) / alpha_1,
                                 coords[lab[i + 1]]]
                        if is_inter(inter, i):
                            n_inter += 1
        else:
            for i in range(0, len(lab) - 2, 2):
                if coords[lab[i]] == coords[lab[i + 2]]:
                    if xa == coords[lab[i]]:
                        inter_in_border = (np.minimum(ya, yb) <=
                                           np.maximum(coords[lab[i + 1]],
                                                      coords[lab[i + 3]])) and \
                                          (np.maximum(ya, yb) >=
                                           np.minimum(coords[lab[i + 1]],
                                                      coords[lab[i + 3]]))
                        if inter_in_border:
                            n_inter += 1
                else:
                    inter = [xa, coords[lab[i + 1]]]
                    if is_inter(inter, i):
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
        coords = self.coord_env
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


# #### TEST

# env = Environment()

# values = np.zeros((90, 60))
# for i in range(90):  # x
#     for j in range(60):  # y
#         values[i, j] = 1*env.is_point_inside(i/10, j/10)

# plt.imshow(values.T, cmap='cool', interpolation='nearest')
# plt.gca().invert_yaxis()
# plt.show()

# assert env.is_segment_inside(1.5, 1.5, 1.5, 4.5)
# assert env.is_segment_inside(1.5, 4.5, 1.5, 1.5)
# assert not env.is_segment_inside(1.5, 6.5, 1.5, 1.5)
