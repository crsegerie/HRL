from hyperparam import Hyperparam
import torch


TensorTorch = type(torch.Tensor())

class HomogeneousZeta:
    """Homogeneous to the state (internal + external) of the agent."""
    def __init__(self, hyperparam: Hyperparam) -> None:
        self.hp = hyperparam
        self.tensor = torch.zeros(self.hp.cst_agent.zeta_shape)

    def get_resource(self, res: int):
        assert res < self.hp.difficulty.n_resources
        return float(self.tensor[self.hp.cst_agent.features_to_index[f"resource_{res}"]])

    def set_resource(self, res: int, val: float):
        assert res < self.hp.difficulty.n_resources
        self.tensor[self.hp.cst_agent.features_to_index[f"resource_{res}"]] = val

    @property
    def muscular_fatigue(self) -> float:
        return float(self.tensor[self.hp.cst_agent.features_to_index["muscular_fatigue"]])

    @muscular_fatigue.setter
    def muscular_fatigue(self, val: float) -> None:
        self.tensor[self.hp.cst_agent.features_to_index["muscular_fatigue"]] = val

    @property
    def sleep_fatigue(self) -> float:
        return float(self.tensor[self.hp.cst_agent.features_to_index["sleep_fatigue"]])

    @sleep_fatigue.setter
    def sleep_fatigue(self, val: float) -> None:
        self.tensor[self.hp.cst_agent.features_to_index["sleep_fatigue"]] = val

    @property
    def x(self) -> float:
        return float(self.tensor[self.hp.cst_agent.features_to_index["x"]])
    
    @x.setter
    def x(self, val: float) -> None:
        self.tensor[self.hp.cst_agent.features_to_index["x"]] = val

    @property
    def y(self) -> float:
        return float(self.tensor[self.hp.cst_agent.features_to_index["y"]])
    
    @y.setter
    def y(self, val: float) -> None:
        self.tensor[self.hp.cst_agent.features_to_index["y"]] = val

    @property
    def homeostatic(self) -> TensorTorch:
        return self.tensor[self.hp.cst_agent.features_to_index["homeostatic"]]

    @homeostatic.setter
    def homeostatic(self, val: TensorTorch) -> None:
        self.tensor[self.hp.cst_agent.features_to_index["homeostatic"]] = val

    @property
    def non_homeostatic(self) -> TensorTorch:
        return self.tensor[self.hp.cst_agent.features_to_index["non_homeostatic"]]

    @non_homeostatic.setter
    def non_homeostatic(self, val: TensorTorch) -> None:
        self.tensor[self.hp.cst_agent.features_to_index["non_homeostatic"]] = val


class Agent:
    def __init__(self, hyperparam: Hyperparam):
        """Initialize the Agent.
        """
        self.hp = hyperparam

        # Setting initial position
        self.zeta = HomogeneousZeta(self.hp)
        self.zeta.x = self.hp.cst_agent.default_pos_x
        self.zeta.y = self.hp.cst_agent.default_pos_y
        self.zeta.muscular_fatigue = self.hp.cst_agent.min_resource
        self.zeta.sleep_fatigue = self.hp.cst_agent.min_resource

    def drive(self, zeta: HomogeneousZeta, epsilon: float = 0.001) -> float:
        """
        Return the Agent's drive which is the distance between the agent's 
        state and the homeostatic set point.
        """
        # in the delta, we only count the internal state.
        # The tree last coordinate do not count in the homeostatic set point.
        delta = zeta.homeostatic
        drive_delta = float(torch.sqrt(epsilon + torch.dot(delta, delta)))
        return drive_delta
        
    def dynamics(self, zeta: HomogeneousZeta, u: HomogeneousZeta) -> HomogeneousZeta:
        """
        Return the Agent's dynamics which is represented by the f function.
        """
        f = HomogeneousZeta(self.hp)
        f.homeostatic = (self.hp.cst_agent.coef_hertz + u.homeostatic) * \
            (zeta.homeostatic + self.hp.cst_agent.x_star)
        f.non_homeostatic = u.non_homeostatic
        return f

    def euler_method(self, zeta: HomogeneousZeta, u: HomogeneousZeta) -> HomogeneousZeta:
        """Euler method for tiny time steps.
        """
        new_zeta = HomogeneousZeta(self.hp)
        delta_zeta = self.hp.cst_algo.time_step * self.dynamics(zeta, u).tensor
        new_zeta.tensor = zeta.tensor + delta_zeta
        return new_zeta

    def integrate_multiple_steps(self,
                                 duration: float,
                                 zeta: HomogeneousZeta,
                                 control: HomogeneousZeta) -> HomogeneousZeta:
        """We integrate rigorously with an exponential over 
        long time period the differential equation.
        This function is usefull in the case of big actions, 
        such as going direclty to one of the resource.
        """
        x = zeta.homeostatic + self.hp.cst_agent.x_star
        rate = self.hp.cst_agent.coef_hertz + control.homeostatic
        new_x = x * torch.exp(rate * duration)
        new_zeta_homeo = new_x - self.hp.cst_agent.x_star

        new_zeta = HomogeneousZeta(self.hp)
        new_zeta.homeostatic = new_zeta_homeo
        new_zeta.non_homeostatic = zeta.non_homeostatic.clone()
        return new_zeta

