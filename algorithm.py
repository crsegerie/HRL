import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import pandas as pd


from agent import Agent, Control, Zeta, Homeostatic, Control
from environment import Environment
from nets import Net_J, Net_f


class Algorithm:
    def __init__(self, env: Environment, agent: Agent, net_J: Net_J, net_f: Net_f):

        # ALGOS METAPARAMETERS ##################################

        # Iterations
        self.N_iter = 100
        self.time_step = 1

        # RL learning
        self.eps = 0.3  # random actions
        self.gamma = 0.99  # discounted rate
        self.tau = 0.001  # not used yet (linked with the target function)

        # Gradient descent
        self.learning_rate = 0.001
        self.asym_coeff = 100

        # plotting
        self.cycle_plot = self.N_iter - 1
        self.N_rolling = 5
        # save neural networks weights every N_save_weights step
        self.N_save_weights = 1000
        self.N_print = 1

        # ACTIONS METAPARAMETERS ##################################

        # We discretized the angles in order no to spin without moving
        # The controls for walking and running for each angle is pre-computed
        num_pos_angles = 5
        self.controls_turn = [[np.cos(2 * np.pi / num_pos_angles * i),
                               np.sin(2 * np.pi / num_pos_angles * i)]
                              for i in range(num_pos_angles)]
        self.controls_turn = torch.Tensor(self.controls_turn)

        self.walking_speed = 0.1
        control_walking = [[0, 0, 0, 0, 0.01, 0,  # homeostatic resources
                            self.walking_speed * self.controls_turn[i][0],  # x
                            self.walking_speed * self.controls_turn[i][1],  # y
                            0]  # angle
                           for i in range(num_pos_angles)]
        self.running_speed = 0.3
        control_running = [[0, 0, 0, 0, 0.05, 0,
                            self.running_speed * self.controls_turn[i][0],
                            self.running_speed * self.controls_turn[i][1],
                            0]
                           for i in range(num_pos_angles)]

        # a list mapping the action to a control.
        # For each action, there is a control verifying for example that
        # the muscular tiredness enables the agent to walk.
        # for example, for the first action (walking), we verify that the
        # tiredness is not above 6.
        # For the second action (running), we verify that the tiredness is
        # not above 3.
        self.actions_controls = [
            # walking
            control_walking,  # -> constraints[0]
            # running
            control_running,  # -> constraints[1]
            # turning an angle to the left
            [0, 0, 0, 0, 0.001, 0, 0, 0, 0],  # etc...
            # turning an angle to the right
            [0, 0, 0, 0, 0.001, 0, 0, 0, 0],
            # sleeping
            [0, 0, 0, 0, 0, -0.001, 0, 0, 0],
            # get resource 1
            [0.1, 0, 0, 0, 0, 0, 0, 0, 0],
            # get resource 2
            [0, 0.1, 0, 0, 0, 0, 0, 0, 0],
            # get resource 3
            [0, 0, 0.1, 0, 0, 0, 0, 0, 0],
            # get resource 4
            [0, 0, 0, 0.1, 0, 0, 0, 0, 0],
            # not doing anything
            [0, 0, 0, 0, 0, 0, 0, 0, 0]]
        # Keep in mind that the agent looses resources and energy even
        # if he does nothing via the function f.
        self.actions_controls = [torch.Tensor(
            x) for x in self.actions_controls]

        # there are 4 additionnal actions : Going to the 4 resource and eating
        self.nb_actions = len(self.actions_controls) + 4

        # a vector representing the physical limits. Example: you cannot eat
        # more than 6 kg of food...
        # Same order of action as actions_controls
        # Constraint verify the tiredness. There are tree types of tiredness:
        # - muscular tiredness (M)
        # - and sleep tiredness (S)
        # - max_food_in_the_stomach (F)
        #                M, M, M, M, S, F,  F,  F,  F,  (*)
        # constraints = [6, 3, 6, 6, 1, 15, 15, 15, 15, None]
        self.constraints = [6, 3, 6, 6, 1, 8, 8, 8, 8, None]
        # (*) There is not any constraint when you do anything.

        # An agent cannot do micro sleep.
        # He cannot wake during a minimum amount of time.
        # ratio of the minimum sleep time for the agent and the time_step
        self.n_min_time_sleep = 1000
        # If the agent is too tired, he automatically sleeps.
        self.max_tired = 10

        self.meaning_actions = {
            0: "walking",
            1: "running",
            2: "turning trigo",
            3: "turning anti trigo",
            4: "sleeping",
            5: "get resource 1",
            6: "get resource 2",
            7: "get resource 3",
            8: "get resource 4",
            9: "not doing anything",
            10: "going direcly to resource 1",
            11: "going direcly to resource 2",
            12: "going direcly to resource 3",
            13: "going direcly to resource 4",
        }

        # If one of the agent resource is lower than min_resource,
        # we put it back at min_resource
        # This help because if one of the agent resource equals 0,
        # because of the dynamics of the exponintial,
        # the agent cannot reconstitute this resource.
        self.min_resource = 0.1

        # CLASSES #########################################
        self.env = env
        self.agent = agent
        self.net_J = net_J
        self.net_f = net_f

        # UTILS ########################################""
        self.optimizer_J = torch.optim.Adam(
            self.net_J.parameters(), lr=self.learning_rate)
        self.optimizer_f = torch.optim.Adam(
            self.net_f.parameters(), lr=self.learning_rate)

        self.historic_zeta = []
        self.historic_actions = []
        self.historic_losses = []  # will contain a list of 2d [L_f, L_J]

    def actions_possible(self):
        """
        Return a list of bool showing which action is permitted or not.

        possible_actions = [
            # walking
            # running
            # turning an angle to the left
            # turning an angle to the right
            # sleeping
            # get resource 1
            # get resource 2
            # get resource 3
            # get resource 4
            # not doing anything
        ]
        + 4 for the action of going to a resource after seeing it
        """
        # There are 4 more possible actions:
        # The 4 last actions correspond to going direcly to each of the
        # 4 ressources if possible.
        possible_actions = [True for i in range(
            len(self.actions_controls) + 4)]

        # If the agent is too tired in his muscle
        if self.agent.zeta[4] >= self.constraints[0]:
            # walking is no more possible
            possible_actions[0] = False

        # He cannot escape the environment when walking.
        x_walk = self.agent.zeta[6] + self.time_step * \
            self.actions_controls[0][int(self.agent.zeta[8])][6]
        y_walk = self.agent.zeta[7] + self.time_step * \
            self.actions_controls[0][int(self.agent.zeta[8])][7]
        if not self.env.is_point_inside(x_walk, y_walk):
            possible_actions[0] = False

        # if the agent is too tired, we cannot run.
        if self.agent.zeta[4] >= self.constraints[1]:
            possible_actions[1] = False

        # He cannot escape the environment when running.
        x_run = self.agent.zeta[6] + self.time_step * \
            self.actions_controls[1][int(self.agent.zeta[8])][6]
        y_run = self.agent.zeta[7] + self.time_step * \
            self.actions_controls[1][int(self.agent.zeta[8])][7]

        if not self.env.is_point_inside(x_run, y_run):
            possible_actions[1] = False

        # He cannot rotate trigonometrically if too tired.
        if self.agent.zeta[4] >= self.constraints[2]:
            possible_actions[2] = False

        # He cannot rotate non-trigonometrically if too tired.
        if self.agent.zeta[4] >= self.constraints[3]:
            possible_actions[3] = False

        # The agent cannot sleep if he is not enouth tired.
        if self.agent.zeta[5] <= self.constraints[4]:
            possible_actions[4] = False

        # If the agent is too sleepy, the only possible action is to sleep.
        if self.agent.zeta[5] >= self.max_tired:
            possible_actions = [False for i in range(
                len(self.actions_controls) + 4)]
            possible_actions[4] = True

        def is_near_ressource(circle: str):
            """circle:  (str) 
                example 'circle_1'"""
            dist = (self.agent.zeta[6] - self.env.coord_circ[circle][0])**2 + (
                self.agent.zeta[7] - self.env.coord_circ[circle][1])**2
            radius = self.env.coord_circ[circle][2]**2
            return dist < radius

        def check_resource(i: int):
            index_circle = 4+i
            if self.agent.zeta[0+i - 1] >= self.constraints[index_circle]:
                possible_actions[index_circle] = False
            if not is_near_ressource(f'circle_{str(i)}'):
                possible_actions[index_circle] = False

        for resource in range(1, 5):
            check_resource(resource)

        def is_resource_visible(resource: int):
            """Check if segment between agent and resource i is visible"""
            xa = self.agent.zeta[6]
            xb = self.env.coord_circ[f'circle_{str(resource)}'][0]
            ya = self.agent.zeta[7]
            yb = self.env.coord_circ[f'circle_{str(resource)}'][1]
            return self.env.is_segment_inside(xa, xb, ya, yb)

        for resource in range(1, 5):
            if not is_resource_visible(resource):
                possible_actions[9+resource] = False

        return possible_actions

    def euler_method(self, u: Control):
        """Euler method for tiny time steps.

        Parameters
        ----------
        u: torch.tensor
            control. (freewill of the agent)

        Returns:
        --------
        The updated zeta.
        """
        delta_zeta = self.time_step * \
            self.agent.dynamics(self.agent.zeta, u)
        return self.agent.zeta + delta_zeta

    def integrate_multiple_steps(self, duration: float, control: Control):
        """We integrate rigorously with an exponential over 
        long time period the differential equation.

        This function is usefull in the case of big actions, 
        such as going direclty to one of the circles.

        PARAMETER:
        ----------
        duration :
            duration of time to integrate over.
        control:
            index of the action.

        RETURNS:
        -------
        new_zeta. The updated zeta."""
        assert len(control) == 6
        assert len(self.agent.zeta[:6]) == 6

        x = self.agent.zeta[:6] + self.agent.x_star
        rate = self.agent.coef_herzt + control
        new_x = x * torch.exp(rate * duration)
        new_zeta = self.agent.zeta.clone()
        new_zeta[:6] = new_x - self.agent.x_star
        return new_zeta

    def going_and_get_resource(self, circle: str):
        """Return the new state associated with the special action a going 
        direclty to the circle.

        Parameter:
        ----------
        circle : example : "circle_1

        Returns:
        --------
        The new state (zeta), but with the agent who has wlaken to go to 
        the state and so which is therefore more tired.
        """
        new_zeta = self.agent.zeta

        agent_x = self.agent.zeta[6]
        agent_y = self.agent.zeta[7]
        circle_x = self.env.coord_circ[circle][0]
        circle_y = self.env.coord_circ[circle][1]
        distance = np.sqrt((agent_x - circle_x)**2 +
                           (agent_y - circle_y)**2)

        if distance != 0:
            # If the agent is at a distance d from the resource,
            # it will first need to walk
            # to consume it. Thus, we integrate the differential
            # equation of its internal state during this time
            time_to_walk = distance * self.time_step / self.walking_speed

            angle = int(self.agent.zeta[8])
            control = self.actions_controls[0][angle][:6]
            new_zeta = self.integrate_multiple_steps(time_to_walk, control)
            new_zeta[6] = self.env.coord_circ[circle][0]
            new_zeta[7] = self.env.coord_circ[circle][1]
            return new_zeta
        else:
            # If the agent is already on the resource, then consuming it is done instantly
            u = self.actions_controls[9]

            self.agent.zeta = self.euler_method(u)

            return new_zeta

    def new_state(self, a: int):
        """Return the new state after an action is taken.

        Parameter:
        ----------
        a: action.
        actions = [
            0# walking
            1# running
            2# turning an angle to the left
            3# turning an angle to the right
            4# sleeping
            5# get resource 1
            6# get resource 2
            7# get resource 3
            8# get resource 4
            9# not doing anything
        ]

        And we have also complementatry actions:
        action_circle = {
            10: "circle_1",
            11: "circle_2",
            12: "circle_3",
            13: "circle_4",
        }


        Returns:
        --------
        The new states.
        """

        # 4 is sleeping
        if a == 4:
            # The new state is when the agent wakes up
            # Therefore, we integrate the differential equation until this time
            duration_sleep = self.n_min_time_sleep * self.time_step
            control_sleep = self.actions_controls[4][:6]
            new_zeta = self.integrate_multiple_steps(
                duration_sleep, control_sleep)

        # going and getting the resource
        elif a in [10, 11, 12, 13]:
            action_circle = {  # TODO: put next to init
                10: "circle_1",
                11: "circle_2",
                12: "circle_3",
                13: "circle_4",
            }
            for a_, circle in action_circle.items():
                if a == a_:
                    new_zeta = self.going_and_get_resource(circle)

        # Other elementary actions
        else:
            u = torch.zeros(self.agent.zeta.shape)

            map_action_control = {
                i: self.actions_controls[i] for i in range(2, 10)}

            map_action_control[0] = self.actions_controls[0][int(
                self.agent.zeta[8])]  # walk
            map_action_control[1] = self.actions_controls[1][int(
                self.agent.zeta[8])]  # run

            u = map_action_control[a]

            # Euler method to calculate the new zeta.
            new_zeta = self.euler_method(u)

            # If the agent is turning its angle
            # Not in euler because discretized.
            if a == 2:
                new_zeta[8] = new_zeta[8] + 1
            elif a == 3:
                new_zeta[8] = new_zeta[8] - 1
            if new_zeta[8] == len(self.actions_controls[0]):
                new_zeta[8] = 0
            elif new_zeta[8] == -1:
                new_zeta[8] = len(self.actions_controls[0]) - 1

        return new_zeta

    def evaluate_action(self, action: int):
        """Return the score associated with the action.

        In this function, we do not seek to update the Net_F and Net_J,
         so we use the eval mode.
        But we still seek to use the derivative of the Net_F according to zeta.
         So we use require_grad = True.
        Generally, inly the parameters of a neural network are on 
        require_grad = True.
        But here we must use zeta.require_grad = True.

        Parameters:
        ----------
        action : int

        Returns:
        --------
        the score : pytorch float.
        """
        # f is a neural network taking one vector.
        # But this vector contains the information of zeta and u.
        # The u is the one-hot-encoded control associated with the action a
        zeta_u = torch.cat(
            [self.agent.zeta, torch.zeros(self.nb_actions)])
        index_control = len(self.agent.zeta) + action
        zeta_u[index_control] = 1

        # Those lines are only used to accelerate the computations but are not
        # strickly necessary.
        # because we don't want to compute the gradient wrt theta_f and theta_J.
        for param in self.net_f.parameters():
            param.requires_grad = False
        for param in self.net_J.parameters():
            param.requires_grad = False

        # In the Hamilton Jacobi Bellman equation, we derivate J by zeta.
        # But we do not want to propagate this gradient.
        # We seek to compute the gradient of J with respect to zeta_to_J.
        self.agent.zeta.requires_grad = True
        # zeta_u_to_f.require_grad = False : This is already the default.

        # Deactivate dropout and batchnorm but continues to accumulate the gradient.
        # This is the reason it is generally used paired with "with torch.no_grad()"
        self.net_J.eval()
        self.net_f.eval()

        # in the no_grad context, all the results of the computations will have
        # requires_grad=False,
        # even if the inputs have requires_grad=True
        # If you want to freeze part of your model and train the rest, you can set
        # requires_grad of the parameters you want to freeze to False.
        f = self.net_f.forward(zeta_u).detach()
        new_zeta_ = self.agent.zeta + self.time_step * f
        instant_reward = self.agent.drive(new_zeta_)
        grad_ = torch.autograd.grad(
            self.net_J(self.agent.zeta), self.agent.zeta)[0]
        future_reward = torch.dot(grad_, self.net_f.forward(zeta_u))
        future_reward = future_reward.detach()

        score = instant_reward + future_reward

        self.agent.zeta.requires_grad = False

        for param in self.net_f.parameters():
            param.requires_grad = True
        for param in self.net_J.parameters():
            param.requires_grad = True

        self.net_J.train()
        self.net_f.train()
        return score.detach().numpy()

    def simulation_one_step(self, k: int):
        """Simulate one step.

        Paramaters:
        -----------
        k: int

        Returns
        -------
        (action, loss): int, np.ndarray"""
        # if you are exacly on 0 (empty resource) you get stuck
        # because of the nature of the differential equation.
        for i in range(6):
            # zeta = x - x_star
            if self.agent.zeta[i] + self.agent.x_star[i] < self.min_resource:
                self.agent.zeta[i] = -self.agent.x_star[i] + self.min_resource

        possible_actions = self.actions_possible()
        indexes_possible_actions = [i for i in range(
            self.nb_actions) if possible_actions[i]]

        # The default action is doing nothing. Like people in real life.
        action = 9

        if np.random.random() <= self.eps:
            action = np.random.choice(indexes_possible_actions)

        else:
            best_score = np.Inf
            for act in indexes_possible_actions:
                score = self.evaluate_action(act)
                if score < best_score:
                    best_score = score
                    action = act

        zeta_u = torch.cat(
            [self.agent.zeta, torch.zeros(self.nb_actions)])
        zeta_u[len(self.agent.zeta) + action] = 1

        new_zeta = self.new_state(action)  # actual choosen new_zeta

        coeff = self.asym_coeff
        # set of big actions leading directly to the resources and 4 is for sleeping
        if action in {4, 10, 11, 12, 13}:
            coeff = 1

        predicted_new_zeta = self.agent.zeta + self.time_step * \
            self.net_f.forward(zeta_u)

        Loss_f = coeff * torch.dot(new_zeta - predicted_new_zeta,
                                   new_zeta - predicted_new_zeta)

        self.optimizer_J.zero_grad()
        self.optimizer_f.zero_grad()
        Loss_f.backward()
        self.optimizer_J.zero_grad()
        self.optimizer_f.step()

        self.agent.zeta.requires_grad = True

        # if drive = d(\zeta_t)= 1 and globally convex environment (instant
        # and long-term improvements are in the same direction)

        # futur drive = d(\zeta_t, u_a) = 0.9
        instant_drive = self.agent.drive(new_zeta)

        # negative
        delta_deviation = torch.dot(torch.autograd.grad(self.net_J(self.agent.zeta),
                                                        self.agent.zeta)[0],
                                    self.net_f.forward(zeta_u))

        # 0.1 current deviation
        discounted_deviation = - torch.log(torch.tensor(self.gamma)) * \
            self.net_J.forward(self.agent.zeta)
        Loss_J = torch.square(
            instant_drive + delta_deviation - discounted_deviation)

        self.agent.zeta.requires_grad = False

        self.optimizer_J.zero_grad()
        self.optimizer_f.zero_grad()
        Loss_J.backward()
        self.optimizer_f.zero_grad()
        self.optimizer_J.step()

        self.agent.zeta = new_zeta

        if (k % self.N_print) == 0:
            print("Iteration:", k, "/", self.N_iter - 1)
            print("Action:", action)
            print("zeta:", self.agent.zeta)
            print("")

        if (k % self.N_save_weights) == 0:
            torch.save(self.net_J.state_dict(), 'weights_net_J')
            torch.save(self.net_f.state_dict(), 'weights_net_f')

        loss = np.array([Loss_f.detach().numpy(), Loss_J.detach().numpy()[0]])
        return action, loss

    def compute_mask(self, scale: int):
        """Compute the mask.

        Parameters:
        -----------
        scale: int

        Returns:
        --------
        is_inside: np.ndarray
        """
        n_X = 9*scale
        n_Y = 6*scale
        values = np.empty((n_X, n_Y))
        values.fill(np.nan)
        is_inside = np.zeros((n_X, n_Y))
        for i in range(n_X):  # x
            for j in range(n_Y):  # y
                is_inside[i, j] = self.env.is_point_inside(i/scale, j/scale)
        return is_inside

    def plot_J(self, ax, fig, resource: int, scale: int, is_inside):
        """Plot of the learned J function.

        Parameters:
        -----------
        ax: SubplotBase
        resource: int
            1, 2, 3, or 4.
        scale: int
            scale squared gives the number of plotted points for a unt square.
        is_inside : np.ndarray

        Returns:
        --------
        None
        """

        self.net_J.eval()

        n_X = 9*scale
        n_Y = 6*scale
        values = np.empty((n_X, n_Y))
        values.fill(np.nan)
        for i in range(n_X):  # x
            for j in range(n_Y):  # y
                if is_inside[i, j]:  # TODO : use torch batch
                    # We are at the optimum for three out of the 4 resources
                    # but one resources varies alongside with the coordinates.
                    # No muscular nor energic fatigues.
                    zeta = torch.Tensor(
                        [0, 0, 0, 0, 0, 0, i/scale, j/scale, 0])
                    zeta[resource-1] = -self.agent.x_star[resource-1]
                    values[i, j] = self.net_J(zeta).detach().numpy()

        im = ax.imshow(X=values.T, cmap="YlGnBu", norm=Normalize())
        ax.axis('off')
        ax.invert_yaxis()
        ax.set_title(f'Deviation function (resource {resource} missing)')
        cbar = fig.colorbar(im, extend='both', shrink=0.4, ax=ax)

    def plot_ressources(self, ax, frame: int):
        """Plot the historic of the ressrouce with time in abscisse.

        Parameters:
        -----------
        ax: SubplotBase
        frame: int

        Returns:
        --------
        None
        Warning : abscisse is not time but step!
        """

        zeta_meaning = [
            "resource_1",
            "resource_2",
            "resource_3",
            "resource_4",
            "muscular energy",
            "aware energy",
            "x",
            "y",
            "angle",
        ]

        df = pd.DataFrame(self.historic_zeta[:frame+1],
                          columns=zeta_meaning)
        df.plot(ax=ax, grid=True, yticks=list(range(0, 10)))  # TODO
        ax.set_ylabel('value')
        ax.set_xlabel('frames')
        ax.set_title("Evolution of the resource")

    def plot_loss(self, ax, frame: int):
        """Plot the loss in order to control the learning of the agent.

        Parameters:
        -----------
        ax: SubplotBase
        frame: int

        Returns:
        --------
        None
        Warning : abscisse is not time but step!
        """

        loss_meaning = [
            "Loss of the transition function $L_f$",
            "Loss of the deviation function $L_J$",
        ]

        df = pd.DataFrame(self.historic_losses[:frame+1],
                          columns=loss_meaning)
        df = df.rolling(window=self.N_rolling).mean()
        df.plot(ax=ax, grid=True, logy=True)
        ax.set_ylabel('value of the losses')
        ax.set_xlabel('frames')
        ax.set_title(
            f"Evolution of the log-loss (moving average with "
            f"{self.N_rolling} frames)")

    def plot_position(self, ax, zeta: Zeta):
        """Plot the position with an arrow.

        Parameters:
        -----------
        ax: SubplotBase
        zeta: torch.tensor

        Returns:
        --------
        None
        Warning : abscisse is not time but step!
        """
        self.env.plot(ax=ax)  # initialisation of plt with background
        x = zeta[6]
        y = zeta[7]

        num_angle = int(zeta[8])

        dx, dy = self.controls_turn[num_angle]

        alpha = 0.5

        ax.arrow(x, y, dx, dy, head_width=0.1, alpha=alpha)
        ax.set_title("Position and orientation of the agent")

    def plot(self, frame: int,  scale=5):
        """Plot the position, angle and the ressources of the agent.

        - time, ressources, historic in transparence 
        -> faire une fonction plot en dehors de l'agent


        Parameters:
        -----------
        frame :int

        Returns:
        --------
        None
        """

        is_inside = self.compute_mask(scale=scale)

        zeta = self.historic_zeta[frame]

        fig = plt.figure(figsize=(16, 16))
        shape = (4, 4)
        ax_resource = plt.subplot2grid(shape, (0, 0), colspan=4)
        ax_env = plt.subplot2grid(shape, (1, 0), colspan=2, rowspan=2)
        ax_loss = plt.subplot2grid(shape, (1, 2), colspan=2, rowspan=2)
        axs_J = [None]*4
        axs_J[0] = plt.subplot2grid(shape, (3, 0))
        axs_J[1] = plt.subplot2grid(shape, (3, 1))
        axs_J[2] = plt.subplot2grid(shape, (3, 2))
        axs_J[3] = plt.subplot2grid(shape, (3, 3))

        last_action = self.historic_actions[frame]

        fig.suptitle(
            (f'Dashboard. Frame: {frame} - last action: '
                f'{last_action}: {self.meaning_actions[last_action]} '),
            fontsize=16)

        self.plot_position(ax=ax_env, zeta=zeta)

        self.plot_ressources(ax=ax_resource, frame=frame)
        self.plot_loss(ax=ax_loss, frame=frame)

        for resource in range(4):
            self.plot_J(ax=axs_J[resource],
                        fig=fig, resource=resource+1,
                        scale=scale, is_inside=is_inside)

        plt.tight_layout()
        name_fig = f"images/frame_{frame}"
        plt.savefig(name_fig)
        print(name_fig)
        plt.close(fig)

    def simulation(self):

        for k in range(self.N_iter):
            print(k)
            action, loss = self.simulation_one_step(k)

            # save historic
            self.historic_zeta.append(self.agent.zeta.detach().numpy())
            self.historic_actions.append(action)
            self.historic_losses.append(loss)

            if k % self.cycle_plot == 0:
                self.plot(k)

        torch.save(self.net_J.state_dict(), 'weights_net_J')
        torch.save(self.net_f.state_dict(), 'weights_net_f')
