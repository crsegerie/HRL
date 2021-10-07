import pandas as pd
from utils import Difficulty

class Action:
    """Defining the possible actions for the agent
    in its environment.
    """

    def __init__(self, difficulty: Difficulty) -> None:
        
        def wlaking_constrain : 
            return None
        
        
        
        actions_list = [
            {"name": "walking_right", "definition": 0, "constraint": wlaking_constrain, "control": 0, "coefficient loss": 0},
            {"name": "walking_left", "definition": 0, "constraint": 0, "control": 0, "coefficient loss": 0},
            {"name": "walking_up", "definition": 0, "constraint": 0, "control": 0, "coefficient loss": 0},
            {"name": "walking_down", "definition": 0, "constraint": 0, "control": 0, "coefficient loss": 0}
            ]
        
        constraints_eating = [lambda (i, x) :  ]
        
        actions_list = actions_list + \
            [{"name": f"eat resource {str(resource)}", "definition": 0, "constraint": lambda , "control": 0, "coefficient loss": 0} 
             for resource in range(difficulty.n_resources)]
        
        
        
        self.actions = pd.DataFrame(actions_list)

        self.n_actions = len(self.actions)
