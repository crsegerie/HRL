import pandas as pd


class Action:
    """Defining the possible actions for the agent
    in its environment.
    """

    def __init__(self) -> None:
        self.actions = pd.DataFrame([
            {"name": "walking_right", "definition": 0, "constraint": 0, "control": 0, "coefficient loss": 0},
            {"name": "walking_left", "definition": 0, "constraint": 0, "control": 0, "coefficient loss": 0},
            {"name": "walking_up", "definition": 0, "constraint": 0, "control": 0, "coefficient loss": 0},
            {"name": "walking_down", "definition": 0, "constraint": 0, "control": 0, "coefficient loss": 0}])

        self.n_actions = len(self.actions)
