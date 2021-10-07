from typing import List
from dataclasses import dataclass

from utils import Difficulty

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


@dataclass
class Point:
    """Point delimiting the boundaries of the environment."""
    x: float
    y: float


@dataclass
class ResourceDC:
    """Resource representing a type of resource in the environment."""
    x: float
    y: float
    r: float
    color: str


class Environment:
    def __init__(self, difficulty : Difficulty):
        # Simulation
        coord_env_polygon = [Point(1, 1),
                             Point(1, 5),
                             Point(2, 5),
                             Point(2, 3),
                             Point(3, 3),
                             Point(3, 4),
                             Point(4, 4),
                             Point(4, 6),
                             Point(9, 6),
                             Point(9, 5),
                             Point(6, 5),
                             Point(6, 3),
                             Point(7, 3),
                             Point(7, 0),
                             Point(6, 0),
                             Point(6, 2),
                             Point(5, 2),
                             Point(5, 1)]

        coord_env_square = [Point(0, 0),
                            Point(10, 0),
                            Point(10, 10),
                            Point(0, 10),]
        
        self.coord_env = coord_env_polygon if difficulty.env == "polygon" else coord_env_square
        
        four_resources: List[ResourceDC] = [
            ResourceDC(x=1.5, y=4.25, r=0.3, color='red'),
            ResourceDC(x=4.5, y=1.5, r=0.3, color='blue'),
            ResourceDC(x=8, y=5.5, r=0.3, color='orange'),
            ResourceDC(x=6.5, y=0.75, r=0.3, color='green'),
        ]

        two_resources: List[ResourceDC] = [
            ResourceDC(x=1.5, y=4.25, r=0.3, color='red'),
            ResourceDC(x=4.5, y=1.5, r=0.3, color='blue'),
        ]
        
        self.resources = two_resources if difficulty.n_resources == 2 else four_resources
        
        self.width = 9 if difficulty.env == "polygon" else 10
        self.height = 6 if difficulty.env == "polygon" else 10

    def is_point_inside(self, x: float, y: float):
        """Check if a point (x,y) is inside the polygon.
        To do this, we look at the number of sides of the polygon at the left
        of the point.
        """

        # It allows no to treat the last case from
        # the end to the beginning separately
        coords = self.coord_env + [self.coord_env[0]]
        n_left = 0

        def is_left(x0, y0, y1):
            cstr_1_y = (y0 > y) and (y1 <= y)
            cstr_2_y = (y0 <= y) and (y1 > y)
            cstr_x = (x0 <= x)
            if (cstr_1_y or cstr_2_y) and cstr_x:
                return True
            return False

        for i, point in enumerate(coords[:-1]):
            if is_left(point.x, point.y, coords[i + 1].y):
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

        # It allows no to treat the last case from
        # the end to the beginning separately
        coords = self.coord_env + [self.coord_env[0]]

        def point_in_seg(point: Point, A: Point, B: Point):
            """Check if a point on the line AB is actually
            on this segment (or just aligned to it)."""
            in_seg = (point.x >= min(A.x, B.x)) and \
                     (point.x <= max(A.x, B.x)) and \
                     (point.y >= min(A.y, B.y)) and \
                     (point.y <= max(A.y, B.y))
            return in_seg

        def is_inter(inter: Point, border0: Point, border1: Point):
            """Check if the intersection between the segment [A, B] 
            and the border number i is both inside [A, B] and the border."""
            inter_in_AB = point_in_seg(inter, Point(x=xa, y=ya), Point(x=xb, y=yb))
            if not inter_in_AB:
                return False
            inter_in_border = point_in_seg(inter, border0, border1)
            if not inter_in_border:
                return False
            return True

        if (xa != xb):
            alpha_1 = (yb - ya) / (xb - xa)
            beta_1 = (ya * xb - yb * xa) / (xb - xa)
            for i, point in enumerate(coords[:-1]):
                if point.x == coords[i + 1].x:
                    inter = Point(x=point.x, y=alpha_1 * point.x + beta_1)
                    if is_inter(inter, point, coords[i + 1]):
                        return False
                else:
                    if ya == yb:
                        if ya == point.y:
                            inter_in_border = (min(xa, xb) <=
                                               max(point.x, coords[i + 1].x)) and \
                                              (max(xa, xb) >=
                                               min(point.x, coords[i + 1].x))
                            if inter_in_border:
                                return False
                    else:
                        inter = Point(x=(point.y - beta_1) / alpha_1, y=point.y)
                        if is_inter(inter, point, coords[i + 1]):
                            return False
        else:
            # xa = xb : usefull when the agent is placed on a resource for example.
            for i, point in enumerate(coords[:-1]):
                if point.x == coords[i + 1].x:
                    if xa == point.x:
                        inter_in_border = (min(ya, yb) <=
                                           max(point.y, coords[i + 1].y)) and \
                                          (max(ya, yb) >=
                                           min(point.y, coords[i + 1].y))
                        if inter_in_border:
                            return False
                else:
                    inter = Point(x=xa, y=point.y)
                    if is_inter(inter, point, coords[i + 1]):
                        return False
        return True

    def plot_resources(self, ax, scale: int, resources: List[int]=[0, 1, 2, 3]):
        """Add circles representing the resources on the plot."""
        for c_i, resource in enumerate(self.resources):
            if c_i in resources:
                x = resource.x * scale
                y = resource.y * scale
                r = resource.r * scale
                color = resource.color
                patch_circle = Circle((x, y), r, color=color)

                ax.add_patch(patch_circle)
                resource_name = f"resource_{c_i}"
                ax.text(x, y, resource_name)

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

        # It allows no to treat the last case from
        # the end to the beginning separately
        coords = self.coord_env + [self.coord_env[0]]

        for i, point in enumerate(coords[:-1]):
            ax.plot([point.x, coords[i + 1].y],
                    [point.y, coords[i + 1].y],
                    '-', color='black', lw=2)

        self.plot_resources(ax, scale=1)

        ax.axis('off')

        if save_fig:
            ax.savefig('environment.eps', format='eps')

