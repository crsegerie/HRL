import os
import random
from typing import Literal
import numpy as np
import torch


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
            self.n_resources: Literal[2, 4] = 4

            self.env: Literal["polygon", "square"] = "square"

        if level == "MEDIUM":
            self.n_resources: Literal[2, 4] = 4

            self.env: Literal["polygon", "square"] = "polygon"
