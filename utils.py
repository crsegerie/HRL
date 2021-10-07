import os
import random
import numpy as np
import torch
from typing import Literal


def set_all_seeds(seed):
    random.seed(seed)
    # os.environ('PYTHONHASHSEED') = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class Difficulty:
    def __init__(self, level: Literal["EASY", "MEDIUM"]):

        if level == "EASY":
            self.n_resources: Literal[2, 4] = 2
            self.env: Literal["polygon", "square"] = "square"

        if level == "MEDIUM":
            self.n_resources: Literal[2, 4] = 4
            self.env: Literal["polygon", "square"] = "polygon"

        def count_actions(self):
            """4 actions for walking
            2*n_resources actions for resources
            (directly consuming or going directly to a resource)
            1 action for sleeping
            1 action for not doing anything"""
            return 6 + 2 * self.n_resources

        self.n_actions = count_actions(self)
