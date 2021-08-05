from numpy.core.fromnumeric import shape
import torch
from math import pi

ZetaT = type(torch.Tensor(shape(9)))
ControlT = type(torch.Tensor(shape(9)))
HomeostaticT = type(torch.Tensor(shape(6)))


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
        zeta : torch.tensor
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

        self.zeta: ZetaT = torch.zeros(9)
        self.zeta[6] = 3  # initialization x position for the agent
        self.zeta[7] = 2  # initialization y position for the agent

    def dynamics(self, zeta: ZetaT, u: ControlT):
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

        f[:6] = self.coef_herzt * (zeta[:6] + self.x_star) + \
            u[:6] * (zeta[:6] + self.x_star)

        # Those coordinates are not homeostatic : they represent the x-speed,
        # y-speed, and angular-speed.
        # The agent can choose himself his speed.
        f[6:9] = u[6:9]
        return f

    def drive(self, zeta: ZetaT, epsilon: float = 0.001):
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
        delta = zeta[:6]
        drive_delta = torch.sqrt(epsilon + torch.dot(delta, delta))
        return drive_delta
