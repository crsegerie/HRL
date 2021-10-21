import os
import random
import numpy as np
import torch

from typing import Literal, List

from dataclasses import dataclass


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class Difficulty:
    def __init__(self, level: Literal["EASY", "MEDIUM"]):
        if level == "EASY":
            self.n_resources: Literal[2, 4] = 2
            self.type_env: Literal["polygon", "square"] = "square"

        if level == "MEDIUM":
            self.n_resources: Literal[2, 4] = 4
            self.type_env: Literal["polygon", "square"] = "polygon"


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

class Cst_env:
    def __init__(self, difficulty: Difficulty):
        coord_env_polygon = [Point(0, 1),
                             Point(0, 5),
                             Point(1, 5),
                             Point(1, 3),
                             Point(2, 3),
                             Point(2, 4),
                             Point(3, 4),
                             Point(3, 6),
                             Point(8, 6),
                             Point(8, 5),
                             Point(5, 5),
                             Point(5, 3),
                             Point(6, 3),
                             Point(6, 0),
                             Point(5, 0),
                             Point(5, 2),
                             Point(4, 2),
                             Point(4, 1)]

        coord_env_square = [Point(0, 0),
                            Point(10, 0),
                            Point(10, 10),
                            Point(0, 10),]
        
        self.coord_env = coord_env_polygon if difficulty.type_env == "polygon" else coord_env_square
        
        four_resources: List[ResourceDC] = [
            ResourceDC(x=0.5, y=4.25, r=0.3, color='red'),
            ResourceDC(x=3.5, y=1.5, r=0.3, color='blue'),
            ResourceDC(x=7, y=5.5, r=0.3, color='orange'),
            ResourceDC(x=5.5, y=0.75, r=0.3, color='green'),
        ]

        two_resources: List[ResourceDC] = [
            ResourceDC(x=0.5, y=4.25, r=0.3, color='red'),
            ResourceDC(x=3.5, y=1.5, r=0.3, color='blue'),
        ]
        
        self.resources = two_resources if difficulty.n_resources == 2 else four_resources
        
        coord_env_x = [point.x for point in self.coord_env]
        coord_env_y = [point.y for point in self.coord_env]

        self.width = np.max(coord_env_x) - np.min(coord_env_x)
        self.height = np.max(coord_env_y) - np.min(coord_env_y)


HomeostaticT = type(torch.Tensor())  # Size 6

class Cst_agent:
    def __init__(self, difficulty: Difficulty):
        self.default_pos_x = 2
        self.default_pos_y = 2

        self.zeta_shape = difficulty.n_resources + 4 # muscular, aware, x, y

        self.walking_speed = 0.1

        # Homeostatic setpoint
        # Resources 1, 2, 3 and 4, muscular fatigue, aware energy
        x_star_4_resources = torch.Tensor([1, 2, 3, 4, 0, 0])
        x_star_2_resources = torch.Tensor([1, 2, 0, 0])
        self.x_star: HomeostaticT = x_star_4_resources \
            if difficulty.n_resources == 4 else x_star_2_resources

        # Parameters of the function f
        # Resources 1, 2, 3 and 4, muscular fatigue, aware energy, x, y
        self.coef_hertz: HomeostaticT = torch.Tensor(
            [-0.05]*difficulty.n_resources +[-0.008, 0.0005])
        

class Cst_actions:
    def __init__(self, difficulty: Difficulty):
        self.n_actions = 6 + 2 * difficulty.n_resources


class Cst_nets:
    def __init__(self):
        self.n_neurons = 128
        self.dropout_rate = 0.15


class Cst_algo:
    def __init__(self):
        self.time_step = 1


class Hyperparam:
    def __init__(self, level: Literal["EASY", "MEDIUM"]):
        self.difficulty = Difficulty(level)
        self.cst_env = Cst_env(self.difficulty)
        self.cst_agent = Cst_agent(self.difficulty)
        self.cst_actions = Cst_actions(self.difficulty)
        self.cst_nets = Cst_nets()
        self.cst_algo = Cst_algo()
