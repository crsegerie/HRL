from hyperparam import Hyperparam, TensorTorch
import torch


class Zeta:
    """State (internal + external) of the agent."""
    def __init__(self, hyperparam: Hyperparam) -> None:
        self.hp = hyperparam
        self.tensor = torch.zeros(self.hp.cst_agent.zeta_shape)

        self.n_homeostatic = self.hp.difficulty.n_resources + 2

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


class Agent:
    def __init__(self, hyperparam: Hyperparam):
        """Initialize the Agent.
        """
        self.hp = hyperparam

        # Setting initial position
        self.zeta = Zeta(self.hp)
        self.zeta.x = self.hp.cst_agent.default_pos_x
        self.zeta.y = self.hp.cst_agent.default_pos_y
        self.zeta.muscular_fatigue = self.hp.cst_agent.min_resource
        self.zeta.sleep_fatigue = self.hp.cst_agent.min_resource

    def drive(self, zeta: Zeta, epsilon: float = 0.001):
        """
        Return the Agent's drive which is the distance between the agent's 
        state and the homeostatic set point.
        ...

        Variables
        ---------
        zeta: torch.tensor
            whole world state.
        u: torch.tensor
            control. (freewill of the agent)
        """
        # in the delta, we only count the internal state.
        # The tree last coordinate do not count in the homeostatic set point.
        delta = zeta.homeostatic
        drive_delta = torch.sqrt(epsilon + torch.dot(delta, delta))
        return drive_delta
        
    def dynamics(self, zeta: Zeta, u: TensorTorch):
        """
        Return the Agent's dynamics which is represented by the f function.

        Variables
        ---------
        zeta: torch.tensor
            whole world state.
        u: torch.tensor
            control. (freewill of the agent)
        """
        f = torch.zeros(self.hp.cst_agent.zeta_shape)
        # Those first coordinates are homeostatic, and with a null control,
        # zeta tends to zero.

        f[:zeta.n_homeostatic] = (self.hp.cst_agent.coef_hertz + u[:zeta.n_homeostatic]) * \
            (zeta.homeostatic + self.hp.cst_agent.x_star)

        # Those coordinates are not homeostatic : they represent the x-speed,
        # y-speed, and angular-speed.
        # The agent can choose himself his speed.
        f[zeta.n_homeostatic:] = u[zeta.n_homeostatic:]
        return f

    def euler_method(self, zeta: Zeta, u: TensorTorch) -> TensorTorch:
        """Euler method for tiny time steps.

        Parameters
        ----------
        zeta: torch.tensor
            whole world state.
        u: torch.tensor
            control. (freewill of the agent)

        Returns:
        --------
        The updated zeta.
        """
        delta_zeta = self.hp.cst_algo.time_step * self.dynamics(zeta, u)
        new_zeta = zeta.tensor + delta_zeta
        return new_zeta

    def integrate_multiple_steps(self, duration: float, zeta: Zeta, control: TensorTorch):
        """We integrate rigorously with an exponential over 
        long time period the differential equation.

        This function is usefull in the case of big actions, 
        such as going direclty to one of the resource.

        PARAMETER:
        ----------
        duration:
            duration of time to integrate over.
        zeta: torch.tensor
            whole world state.
        u: torch.tensor
            control. (freewill of the agent)

        RETURNS:
        -------
        new_zeta. The updated zeta."""
        x = zeta.homeostatic + self.hp.cst_agent.x_star
        rate = self.hp.cst_agent.coef_hertz + control[:zeta.n_homeostatic]
        new_x = x * torch.exp(rate * duration)
        new_zeta = zeta.tensor.clone()
        new_zeta[:zeta.n_homeostatic] = new_x - self.hp.cst_agent.x_star
        return new_zeta

