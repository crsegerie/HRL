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
  
  
difficultyT = Literal["EASY", "MEDIUM"] 