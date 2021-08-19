from numpy.core.fromnumeric import shape
import torch
from math import pi

ZetaTensorT = type(torch.Tensor(shape(9)))
ControlT = type(torch.Tensor(shape(9)))
HomeostaticT = type(torch.Tensor(shape(6)))



class Zeta:
    """Internal and external state of the agent."""

    def __init__(self, x = None, y = None) -> None:
        """zeta : torch.tensor
        zeta[0] : resource 0
        zeta[1] : resource 1
        zeta[2] : resource 2
        zeta[3] : resource 3
        zeta[4] : muscular energy (muscular resource)
        zeta[5] : aware energy (aware resource) : low if sleepy.
        zeta[6] : x-coordinate
        zeta[7] : y-coordinate
        zeta[8] : angle

        Be aware that zeta[:6] is homeostatic and that zeta[6:] aren't.    
        """
        self._zeta_tensor : ZetaTensorT = torch.zeros(9)
        if x and y:
            self._zeta_tensor[6] = x
            self._zeta_tensor[7] = y
        self.shape = 9
    

    def resource(self, resource_i : int):
        assert resource_i < 4
        return self._zeta_tensor[resource_i]
    
    
    @property
    def muscular_energy(self):
        return self._zeta_tensor[4]

    @property
    def aware_energy(self):
        return self._zeta_tensor[5]
    
    @property
    def x(self):
        return self._zeta_tensor[6]
    
    @property
    def y(self):
        return self._zeta_tensor[7]

    @property
    def angle(self):
        return self._zeta_tensor[8]
    
    @angle.setter
    def angle(self, value):
        self._zeta_tensor[8] = value

    @property
    def homeostatic(self):
        """Homeostatic level is regulated to a set point."""
        return self._zeta_tensor[:6]

    @property
    def position(self):
        """Position coordinates are regulated to a set point."""
        return self._zeta_tensor[6:]


class Agent:
    def __init__(self):
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
        angle_visual_field : float
            in radiant. Not implemented.
        zeta: state of the agent.

        """
        # METAPARAMETERS ##########################################

        # homeostatic point
        # Resources 1, 2, 3, 4 and muscular fatigues and sleep fatigue
        self.x_star: HomeostaticT = torch.Tensor([1, 2, 3, 4, 0, 0])

        # parameters of the function f
        # same + x, y, and angle coordinates
        self.coef_herzt: HomeostaticT = torch.Tensor(
            [-0.05, -0.05, -0.05, -0.05, -0.008, 0.0005])

        # Not used currently
        self.angle_visual_field = pi / 10

        # UTILS ##################################################
        
        # Setting initial position
        # Btw, at the begining the agent is starving and exhausted...
        self.zeta: Zeta = Zeta(x=3, y=2)


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
        # Those first coordinate are homeostatic, and with a null control,
        # zeta tends to zero.

        f[:6] = self.coef_herzt * (zeta.homeostatic + self.x_star) + \
            u[:6] * (zeta.homeostatic + self.x_star)

        # Those coordinates are not homeostatic : they represent the x-speed,
        # y-speed, and angular-speed.
        # The agent can choose himself his speed.
        f[6:9] = u[6:9]
        return f

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
