from typing import Any, Dict, Literal, List
from utils import Difficulty
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import pandas as pd


from agent import Agent, ControlT, ZetaTensorT, HomeostaticT, ControlT, Zeta
from environment import Environment
from nets import Net_J, Net_f


class Algorithm:
    def __init__(self, difficulty: Difficulty, env: Environment, agent: Agent, net_J: Net_J, net_f: Net_f):

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

        self.walking_speed = 0.1

        n_resources = difficulty.n_resources
        n_shape = agent.zeta.shape
        
        # a list mapping the action to a control.
        # For each action, there is a control verifying for example that
        # the muscular tiredness enables the agent to walk.
        # for example, for the first action (walking), we verify that the
        # tiredness is not above 6.
        # For the second action (running), we verify that the tiredness is
        # not above 3.

        self.actions_controls: Dict[str, Any] = {
            "walking_right": [0.]*n_resources + [0.01, 0., self.walking_speed, 0.], 
            "walking_left": [0.]*n_resources + [0.01, 0., -self.walking_speed, 0.], 
            "walking_up": [0.]*n_resources + [0.01, 0., 0., self.walking_speed], 
            "walking_down": [0.]*n_resources + [0.01, 0., 0., -self.walking_speed], 
            "sleeping": [0.]*n_resources + [0., -0.001, 0., 0.],
        }

        # Simply eating
        for resource in range(n_resources):
            control = [0.]*n_shape
            control[resource] = 0.1
            self.actions_controls[f"eat resource {str(resource)}"] = control

        self.actions_controls["not doing anything"] = [0.]*n_shape

        # Keep in mind that the agent looses resources and energy even
        # if he does nothing via the function f.
        self.actions_controls = {
            key: torch.Tensor(x)
            for key, x in self.actions_controls.items()}

        # there are 4 additionnal actions : Going to the 4 resource and eating
        self.nb_actions = len(self.actions_controls) + n_resources

        # a vector representing the physical limits. Example: you cannot eat
        # more than 6 kg of food...
        # Same order of action as actions_controls
        # Constraint verify the tiredness. There are tree types of tiredness:
        # - muscular tiredness (M)
        # - and sleep tiredness (S)
        # - max_food_in_the_stomach (F)
        #               walkR  L  U  D Sleep, eat
        #                   M, M, M, M, S,    F*n_resources     (*)
        # TODO:: dans pandas
        self.constraints = [6, 6, 6, 6, 1] + [8]*n_resources + [None]
        # (*) There is not any constraint when you do anything.

        # An agent cannot do micro sleep.
        # He cannot wake during a minimum amount of time.
        # ratio of the minimum sleep time for the agent and the time_step
        self.n_min_time_sleep = 1000
        # If the agent is too tired, he automatically sleeps.
        self.max_tired = 10

        self.meaning_actions = {
            i: key for i, key
            in enumerate(self.actions_controls.keys())}

        meaning_big_actions = {
            len(self.actions_controls) + r: f"going direcly to resource {str(r)}"
            for r in range(n_resources)
        }

        self.meaning_actions.update(meaning_big_actions)
        self.inv_meaning_actions = {v: k for k,
                                    v in self.meaning_actions.items()}

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

        # UTILS ############################################
        self.difficulty = difficulty
        self.optimizer_J = torch.optim.Adam(
            self.net_J.parameters(), lr=self.learning_rate)
        self.optimizer_f = torch.optim.Adam(
            self.net_f.parameters(), lr=self.learning_rate)

        self.historic_zeta: List[Zeta] = []
        self.historic_actions = []
        self.historic_losses = []  # will contain a list of 2d [L_f, L_J]

    def actions_possible(self):
        """Return a list of bool showing which action is permitted or not.

        + 4 for the action of going to a resource after seeing it
        """
        zeta = self.agent.zeta

        # TODO:: boucle sur pandas
        # There are 4 more possible actions:
        # The 4 last actions correspond to going direcly to each of the
        # 4 ressources if possible.
        possible_actions = [True for i in range(
            len(self.actions_controls) + self.agent.zeta.shape)]

        # If the agent is too tired in his muscle
        if zeta.muscular_energy >= self.constraints[0]:
            # walking is no more possible
            possible_actions[self.inv_meaning_actions["walking_right"]] = False
            possible_actions[self.inv_meaning_actions["walking_left"]] = False
            possible_actions[self.inv_meaning_actions["walking_up"]] = False
            possible_actions[self.inv_meaning_actions["walking_down"]] = False

        # He cannot escape the environment when walking.
        # TODO: zeta_meaning["x"] = 6
        # TODO: x_indice = 6
        str_walkings = ["walking_up", "walking_down", "walking_right", "walking_left"]
        for str_walking in str_walkings:
            x_walk = zeta.x + self.time_step * \
                self.actions_controls[str_walking][self.agent.zeta.x_indice]
            y_walk = zeta.y + self.time_step * \
                self.actions_controls[str_walking][self.agent.zeta.y_indice]
            x_walk, y_walk = float(x_walk), float(y_walk)
            if not self.env.is_point_inside(x_walk, y_walk):
                possible_actions[self.inv_meaning_actions[str_walking]] = False


        # The agent cannot sleep if he is not enouth tired.
        if zeta.aware_energy <= self.constraints[self.inv_meaning_actions["sleeping"]]:
            possible_actions[self.inv_meaning_actions["sleeping"]] = False

        # If the agent is too sleepy, the only possible action is to sleep.
        if zeta.aware_energy >= self.max_tired:
            possible_actions = [False for p in possible_actions]
            possible_actions[self.inv_meaning_actions["sleeping"]] = True

        def is_near_resource(resource_i: int):
            dist = (zeta.x - self.env.resources[resource_i].x)**2 + (
                zeta.y - self.env.resources[resource_i].y)**2
            radius = self.env.resources[resource_i].r**2
            return dist < radius

        def check_resource_eatable(resource_i: int):
            # 4 because there are walking, running, turning trigo and turning anti trigo
            index_resource = self.inv_meaning_actions[f"eat resource {resource_i}"]

            # It cannont eat if he hax already "le ventre plein"
            if zeta.resource(resource_i) >= self.constraints[index_resource]:
                possible_actions[index_resource] = False

            # It cannot eat if he too far away.
            if not is_near_resource(resource_i):
                possible_actions[index_resource] = False

        n_resources = self.agent.zeta.difficulty.n_resources
        for resource in range(n_resources):
            check_resource_eatable(resource)

        def is_resource_visible(resource: int):
            """Check if segment between agent and resource i is visible"""
            xa = float(zeta.x)
            xb = self.env.resources[resource].x
            ya = float(zeta.y)
            yb = self.env.resources[resource].y
            return self.env.is_segment_inside(xa, xb, ya, yb)

        # big actions : seing a resource and eat it.
        for resource in range(n_resources):
            if not is_resource_visible(resource):
                possible_actions[len(possible_actions) -
                                 n_resources + resource] = False

        return possible_actions

    def euler_method(self, u: ControlT) -> ZetaTensorT:
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
        return self.agent.zeta.tensor + delta_zeta

    def integrate_multiple_steps(self, duration: float, control: ControlT):
        """We integrate rigorously with an exponential over 
        long time period the differential equation.

        This function is usefull in the case of big actions, 
        such as going direclty to one of the resource.

        PARAMETER:
        ----------
        duration:
            duration of time to integrate over.
        control:
            index of the action.

        RETURNS:
        -------
        new_zeta. The updated zeta."""
        x = self.agent.zeta.homeostatic + self.agent.x_star
        rate = self.agent.coef_hertz + control
        new_x = x * torch.exp(rate * duration)
        new_zeta = self.agent.zeta.tensor.clone()
        new_zeta[:self.agent.zeta.n_homeostatic] = new_x - self.agent.x_star
        return new_zeta

    def going_and_get_resource(self, resource_i: int) -> ZetaTensorT:
        """Return the new state associated with the special action a going 
        direclty to the resource.

        Parameter:
        ----------
        resource : example : "resource_1

        Returns:
        --------
        The new state (zeta), but with the agent who has wlaken to go to 
        the state and so which is therefore more tired.
        """
        new_zeta_tensor = self.agent.zeta.tensor

        agent_x = self.agent.zeta.x
        agent_y = self.agent.zeta.y
        resource_x = self.env.resources[resource_i].x
        resource_y = self.env.resources[resource_i].y
        distance = np.sqrt((agent_x - resource_x)**2 +
                           (agent_y - resource_y)**2)

        if distance == 0:
            # If the agent is already on the resource, then consuming it is done instantly
            u = self.actions_controls["not doing anything"]

            self.agent.zeta.tensor = self.euler_method(u)

            return new_zeta_tensor

        # If the agent is at a distance d from the resource,
        # it will first need to walk
        # to consume it. Thus, we integrate the differential
        # equation of its internal state during this time
        time_to_walk = distance * self.time_step / self.walking_speed
        
        # We take only homeostatic part of the control
        # So the direction does not matter here
        # TODO: LI norm
        control = self.actions_controls["walking_up"][:self.agent.zeta.n_homeostatic]
        new_zeta_tensor = self.integrate_multiple_steps(
            time_to_walk, control)

        new_zeta_tensor[self.agent.zeta.x_indice] = self.env.resources[resource_i].x
        new_zeta_tensor[self.agent.zeta.y_indice] = self.env.resources[resource_i].y
        return new_zeta_tensor
            

    def new_state(self, a: int) -> ZetaTensorT:
        """Return the new state after an action is taken.

        Parameter:
        ----------
        a: action.
        actions = [
            0# walking right
            1# walking left
            2# walking up
            3# walking down
            4# sleeping
            5# eat 0
            6# eat 1
            7# eat 2
            8# eat 3
            9# not doing anything
        ]

        And we have also complementatry actions:
        action_resource = {
            10: "resource_0",
            11: "resource_1",
            12: "resource_2",
            13: "resource_3",
        }


        Returns:
        --------
        The new states.
        """
        new_zeta = Zeta(self.difficulty)

        # 4 is sleeping
        if a == self.inv_meaning_actions["sleeping"]:
            # The new state is when the agent wakes up
            # Therefore, we integrate the differential equation until this time
            duration_sleep = self.n_min_time_sleep * self.time_step
            control_sleep = self.actions_controls["sleeping"][:self.agent.zeta.n_homeostatic]
            new_zeta.tensor = self.integrate_multiple_steps(
                duration_sleep, control_sleep)

        # going direcly to resource
        elif a in [self.inv_meaning_actions[f"going direcly to resource {i}"]
                   for i in range(self.difficulty.n_resources)]:
            resource_i = int(self.meaning_actions[a][-1:])
            self.going_and_get_resource(resource_i)

        # Other actions: elementary actions
        else:
            u = torch.zeros(self.agent.zeta.shape)
            a_meaning = list(self.actions_controls.keys())[a]
            u = self.actions_controls[a_meaning]

            # Euler method to calculate the new zeta.
            new_zeta.tensor = self.euler_method(u)


        return new_zeta.tensor

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
        _zeta_tensor = self.agent.zeta.tensor
        # f is a neural network taking one vector.
        # But this vector contains the information of zeta and u.
        # The u is the one-hot-encoded control associated with the action a
        zeta_u = torch.cat(
            [_zeta_tensor, torch.zeros(self.nb_actions)])
        index_control = len(_zeta_tensor) + action
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
        _zeta_tensor.requires_grad = True
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
        new_zeta_tensor = _zeta_tensor + self.time_step * f
        new_zeta = Zeta(self.difficulty)
        new_zeta.tensor = new_zeta_tensor
        instant_reward = self.agent.drive(new_zeta)
        grad_ = torch.autograd.grad(
            self.net_J(_zeta_tensor), _zeta_tensor)[0]
        future_reward = torch.dot(grad_, self.net_f.forward(zeta_u))
        future_reward = future_reward.detach()

        score = instant_reward + future_reward

        _zeta_tensor.requires_grad = False
        self.agent.zeta.tensor = _zeta_tensor

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
        _zeta = self.agent.zeta.tensor
        # if you are exacly on 0 (empty resource) you get stuck
        # because of the nature of the differential equation.

        for i in range(self.agent.zeta.n_homeostatic):
            # zeta = x - x_star
            if _zeta[i] + self.agent.x_star[i] < self.min_resource:
                _zeta[i] = -self.agent.x_star[i] + self.min_resource

        possible_actions = self.actions_possible()
        indexes_possible_actions = [i for i in range(
            self.nb_actions) if possible_actions[i]]

        # The default action is doing nothing. Like people in real life.
        action = self.inv_meaning_actions["not doing anything"]

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
            [_zeta, torch.zeros(self.nb_actions)])
        zeta_u[len(_zeta) + action] = 1

        _new_zeta = self.new_state(action)  # actual choosen new_zeta
        
        if self.meaning_actions[action] in ["walking_right", "walking_up", "walking_down", "walking_left"]:
            self.agent.zeta.last_direction = self.meaning_actions[action]
        

        coeff = self.asym_coeff
        # set of big actions leading directly to the resources and 4 is for sleeping
        indices_going_directly = [
            self.inv_meaning_actions[f"going direcly to resource {i}"]
            for i in range(self.difficulty.n_resources)]
        if action in [self.inv_meaning_actions["sleeping"]] + indices_going_directly:
            coeff = 1

        predicted_new_zeta = _zeta + self.time_step * \
            self.net_f.forward(zeta_u)

        Loss_f = coeff * torch.dot(_new_zeta - predicted_new_zeta,
                                   _new_zeta - predicted_new_zeta)

        self.optimizer_J.zero_grad()
        self.optimizer_f.zero_grad()
        Loss_f.backward()
        self.optimizer_J.zero_grad()
        self.optimizer_f.step()

        _zeta.requires_grad = True

        # if drive = d(\zeta_t)= 1 and globally convex environment (instant
        # and long-term improvements are in the same direction)

        # futur drive = d(\zeta_t, u_a) = 0.9
        new_zeta = Zeta(self.difficulty)
        new_zeta.tensor = _new_zeta
        instant_drive = self.agent.drive(new_zeta)

        # negative
        delta_deviation = torch.dot(torch.autograd.grad(self.net_J(_zeta),
                                                        _zeta)[0],
                                    self.net_f.forward(zeta_u))

        # 0.1 current deviation
        discounted_deviation = - torch.log(torch.tensor(self.gamma)) * \
            self.net_J.forward(_zeta)
        Loss_J = torch.square(
            instant_drive + delta_deviation - discounted_deviation)

        _zeta.requires_grad = False

        self.optimizer_J.zero_grad()
        self.optimizer_f.zero_grad()
        Loss_J.backward()
        self.optimizer_f.zero_grad()
        self.optimizer_J.step()

        self.agent.zeta.tensor = _new_zeta

        if (k % self.N_print) == 0:
            print("Iteration:", k, "/", self.N_iter - 1)
            print("Action:", action, self.meaning_actions[action])
            print("zeta:", _zeta)
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
        n_X = self.env.width*scale
        n_Y = self.env.height*scale
        values = np.empty((n_X, n_Y))
        values.fill(np.nan)
        is_inside = np.zeros((n_X, n_Y))
        for i in range(n_X):  # x
            for j in range(n_Y):  # y
                is_inside[i, j] = self.env.is_point_inside(i/scale, j/scale)
        return is_inside

    def plot_J(self, ax, fig, resource_id: int, scale: int, is_inside):
        """Plot of the learned J function.

        Parameters:
        -----------
        ax: SubplotBase
        resource_i: int
        scale: int
            scale squared gives the number of plotted points for a unt square.
        is_inside : np.ndarray

        Returns:
        --------
        None
        """

        self.net_J.eval()

        n_X = self.env.width * scale
        n_Y = self.env.height * scale
        values = np.empty((n_X, n_Y))
        values.fill(np.nan)
        # We could optimize this plot by using a batch with each element of
        # the batch representing one pixel in the image.
        # But this function represents only 1/8 of the total execution time.
        for i in range(n_X):  # x
            for j in range(n_Y):  # y
                if is_inside[i, j]:  # TODO : use torch batch
                    # We are at the optimum for three out of the 4 resources
                    # but one resources varies alongside with the coordinates.
                    # No muscular nor energic fatigues.
                    zeta = torch.Tensor(
                        [0.] * self.difficulty.n_resources + [0., 0., i/scale, j/scale])
                    zeta[resource_id] = -self.agent.x_star[resource_id]
                    values[i, j] = self.net_J(zeta).detach().numpy()

        im = ax.imshow(X=values.T, cmap="YlGnBu", norm=Normalize())
        ax.axis('off')
        ax.invert_yaxis()

        self.env.plot_resources(ax, scale, resources=[resource_id])

        ax.set_title(f'Deviation function (resource {resource_id} missing)')
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

        zeta_meaning = [f"resource_{i}" for i in range(self.difficulty.n_resources)] + \
            [
            "muscular energy",
            "aware energy",
            "x",
            "y",
        ]

        historic_zeta_tensor = [zeta.tensor.detach(
        ).numpy() for zeta in self.historic_zeta[:frame+1]]

        df = pd.DataFrame(historic_zeta_tensor, columns=zeta_meaning)
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
        """Plot the position.

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
        x = zeta.x
        y = zeta.y

        dx, dy = 0, 0
        if self.agent.zeta.last_direction == "walking_right":
            dx, dy = 1, 0
        if self.agent.zeta.last_direction == "walking_left":
            dx, dy = -1, 0
        if self.agent.zeta.last_direction == "walking_up":
            dx, dy = 0, 1
        if self.agent.zeta.last_direction == "walking_down":
            dx, dy = 0, -1
            

        alpha = 0.5

        ax.arrow(x, y, dx, dy, head_width=0.1, alpha=alpha)
        ax.set_title("Position of the agent.")

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

        axs_J = [None]*self.difficulty.n_resources

        for resource in range(self.difficulty.n_resources):
            axs_J[resource] = plt.subplot2grid(shape, (3, resource))

        last_action = self.historic_actions[frame]

        fig.suptitle(
            (f'Dashboard. Frame: {frame} - last action: '
                f'{last_action}: {self.meaning_actions[last_action]} '),
            fontsize=16)

        self.plot_position(ax=ax_env, zeta=zeta)

        self.plot_ressources(ax=ax_resource, frame=frame)
        self.plot_loss(ax=ax_loss, frame=frame)

        for resource_id in range(self.difficulty.n_resources):
            self.plot_J(ax=axs_J[resource_id],
                        fig=fig, resource_id=resource_id,
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
            self.historic_zeta.append(self.agent.zeta)
            self.historic_actions.append(action)
            self.historic_losses.append(loss)

            if k % self.cycle_plot == 0:
                self.plot(k)

        torch.save(self.net_J.state_dict(), 'weights_net_J')
        torch.save(self.net_f.state_dict(), 'weights_net_f')
