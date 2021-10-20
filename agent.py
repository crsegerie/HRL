from utils import Difficulty
import torch

ZetaTensorT = type(torch.Tensor())  # Size 8
ControlT = type(torch.Tensor())  # Size 8
HomeostaticT = type(torch.Tensor())  # Size 6


class Zeta:
    """Internal and external state of the agent."""

    def __init__(self, difficulty: Difficulty, x=None, y=None) -> None:
        """zeta : torch.tensor
        
        homeostatic:
        zeta[0] : resource 0
        zeta[1] : resource 1
        zeta[2] : resource 2
        zeta[3] : resource 3
        zeta[4] : muscular energy (muscular resource)
        zeta[5] : aware energy (aware resource) : high if sleepy.
        
        not homeostatic (position)
        zeta[6] : x-coordinate
        zeta[7] : y-coordinate

        Be aware that zeta[:6] is homeostatic and that zeta[6:] aren't.    
        """
        self.difficulty: Difficulty = difficulty

        self.n_homeostatic = self.difficulty.n_resources + 2 # muscle, aware
        self.shape = self.n_homeostatic + 2 # for both coordinates x, y
        self.tensor: ZetaTensorT = torch.zeros(self.shape)
        self.x_indice = self.n_homeostatic + 0
        self.y_indice = self.n_homeostatic + 1
        self.last_direction = "none"

        if x and y:
            self.tensor[self.n_homeostatic + 0] = x
            self.tensor[self.n_homeostatic + 1] = y
            

    def resource(self, resource_i: int):
        assert resource_i < self.difficulty.n_resources
        return float(self.tensor[resource_i])

    @property
    def muscular_fatigue(self) -> float:
        return float(self.tensor[self.difficulty.n_resources + 0])

    @property
    def aware_energy(self) -> float:
        return float(self.tensor[self.difficulty.n_resources + 1])

    @property
    def x(self) -> float:
        return float(self.tensor[self.n_homeostatic + 0])
    
    @x.setter
    def x(self, x : float) -> None:
        self.tensor[self.n_homeostatic + 0] = x

    @property
    def y(self) -> float:
        return float(self.tensor[self.n_homeostatic + 1])
    
    @y.setter
    def y(self, y : float) -> None:
        self.tensor[self.n_homeostatic + 1] = y

    @property
    def homeostatic(self):
        """Homeostatic level is regulated to a set point."""
        return self.tensor[:self.n_homeostatic]

    @property
    def position(self):
        """Position coordinates are regulated to a set point."""
        return self.tensor[self.n_homeostatic:]


class Agent:
    def __init__(self, difficulty: Difficulty):
        """Initialize the Agent.

        ...

        PROPERTIES
        ---------
        x_star : torch.tensor
            homeostatic set point.
        c : torch.tensor
            homogeneus to the inverse of a second. For example c = (-0.1, ...)
            says that the half-life (like a radioactive element) of the first
            ressource is equal to 10 seconds.
        zeta: state of the agent.

        """
        # METAPARAMETERS ##########################################

        self.time_step = 1
        self.walking_speed = 0.1

        # homeostatic point
        # Resources 1, 2, 3, 4 and muscular fatigues and sleep fatigue
        x_star_4_resources = torch.Tensor([1, 2, 3, 4, 0, 0])
        x_star_2_resources = torch.Tensor([1, 2, 0, 0])

        self.x_star: HomeostaticT = x_star_4_resources \
            if difficulty.n_resources == 4 else x_star_2_resources

        # parameters of the function f
        # same + x, y
        self.coef_hertz: HomeostaticT = torch.Tensor(
            [-0.05]*difficulty.n_resources +[-0.008, 0.0005])

        # UTILS ##################################################

        # Setting initial position
        self.zeta: Zeta = Zeta(difficulty=difficulty, x=3, y=2)

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
        
    def dynamics(self, zeta: Zeta, u: ControlT):
        """
        Return the Agent's dynamics which is represented by the f function.

        Variables
        ---------
        zeta: torch.tensor
            whole world state.
        u: torch.tensor
            control. (freewill of the agent)
        """
        f = torch.zeros(zeta.shape)
        # Those first coordinates are homeostatic, and with a null control,
        # zeta tends to zero.

        f[:zeta.n_homeostatic] = self.coef_hertz * (zeta.homeostatic + self.x_star) + \
            u[:zeta.n_homeostatic] * (zeta.homeostatic + self.x_star)

        # Those coordinates are not homeostatic : they represent the x-speed,
        # y-speed, and angular-speed.
        # The agent can choose himself his speed.
        f[zeta.n_homeostatic:] = u[zeta.n_homeostatic:]
        return f

    def euler_method(self, zeta: Zeta, u: ControlT) -> ZetaTensorT:
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
        delta_zeta = self.time_step * self.dynamics(zeta, u)
        new_zeta = zeta.tensor + delta_zeta
        return new_zeta

    def integrate_multiple_steps(self, duration: float, zeta: Zeta, control: ControlT):
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
        x = zeta.homeostatic + self.x_star
        rate = self.coef_hertz + control[:zeta.n_homeostatic]
        new_x = x * torch.exp(rate * duration)
        new_zeta = zeta.tensor.clone()
        new_zeta[:zeta.n_homeostatic] = new_x - self.x_star
        return new_zeta

