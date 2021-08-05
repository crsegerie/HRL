from typing import List
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np


from dataclasses import dataclass


@dataclass
class CircleDC:
    """Circle representing a type of resource in the environmeet."""

    x: float
    y: float
    r: float
    color: str


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

        self.circles: List[CircleDC] = [
            CircleDC(x=1.5, y=4.25, r=0.3, color='red'),
            CircleDC(x=4.5, y=1.5, r=0.3, color='blue'),
            CircleDC(x=8, y=5.5, r=0.3, color='orange'),
            CircleDC(x=6.5, y=0.75, r=0.3, color='green'),
        ]

    def is_point_inside(self, x: float, y: float):
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

    def is_segment_inside(self, xa: float, xb: float, ya: float, yb: float):
        """Check if the segment AB with A(xa, ya) and B(xb, yb) is completely
        inside the polygon.
        To do this, we look at the number of intersections between the segment
        and the sides of the polygon.
        """

        coords = self.coord_env
        lab = list(coords.keys()) + list(coords.keys())[:2]

        def is_inter(inter: List[float], ind: int):
            """Check if the intersection between the segement [A, B] 
            and the border number i is both inside [A, B] and the border."""
            inter_in_AB = (inter[0] >= min(xa, xb)) and \
                          (inter[0] <= max(xa, xb)) and \
                          (inter[1] >= min(ya, yb)) and \
                          (inter[1] <= max(ya, yb))
            if not inter_in_AB:
                return False
            inter_in_border = (inter[0] >= min(coords[lab[ind]],
                                               coords[lab[ind + 2]])) and \
                (inter[0] <= max(coords[lab[ind]],
                                 coords[lab[ind + 2]])) and \
                (inter[1] >= min(coords[lab[ind + 1]],
                                 coords[lab[ind + 3]])) and \
                (inter[1] <= max(coords[lab[ind + 1]],
                                 coords[lab[ind + 3]]))

            if not inter_in_border:
                return False
            return True

        if (xa != xb):
            alpha_1 = (yb - ya) / (xb - xa)
            beta_1 = (ya * xb - yb * xa) / (xb - xa)
            for i in range(0, len(lab) - 2, 2):
                if coords[lab[i]] == coords[lab[i + 2]]:
                    inter = [coords[lab[i]], alpha_1 * coords[lab[i]] + beta_1]
                    if is_inter(inter, i):
                        return False
                else:
                    if ya == yb:
                        if ya == coords[lab[i + 1]]:
                            inter_in_border = (min(xa, xb) <=
                                               max(coords[lab[i]],
                                                   coords[lab[i + 2]])) and \
                                              (max(xa, xb) >=
                                               min(coords[lab[i]],
                                                   coords[lab[i + 2]]))
                            if inter_in_border:
                                return False
                    else:
                        inter = [(coords[lab[i + 1]] - beta_1) / alpha_1,
                                 coords[lab[i + 1]]]
                        if is_inter(inter, i):
                            return False
        else:
            # xa = xb : usefull when the agent is place on a resource for example.
            for i in range(0, len(lab) - 2, 2):
                if coords[lab[i]] == coords[lab[i + 2]]:
                    if xa == coords[lab[i]]:
                        inter_in_border = (min(ya, yb) <=
                                           max(coords[lab[i + 1]],
                                               coords[lab[i + 3]])) and \
                                          (max(ya, yb) >=
                                           min(coords[lab[i + 1]],
                                               coords[lab[i + 3]]))
                        if inter_in_border:
                            return False
                else:
                    inter = [xa, coords[lab[i + 1]]]
                    if is_inter(inter, i):
                        return False
        return True

    def plot_circles(self, ax, scale: int, circles: List[int]=[0, 1, 2, 3]):
        """Add on the axis the circles representing the resources."""
        for c_i, circle in enumerate(self.circles):
            if c_i in circles:
                x = circle.x * scale
                y = circle.y * scale
                r = circle.r * scale
                color = circle.color
                patch_circle = Circle((x, y), r, color=color)

                ax.add_patch(patch_circle)
                circle_name = f"circle_{c_i}"
                ax.text(x, y, circle_name)

    def plot(self, ax=None, save_fig: bool = False):
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

        self.plot_circles(ax, scale=1)

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
