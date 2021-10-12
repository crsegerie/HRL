from typing import Any, Dict, Literal, List
from utils import Difficulty
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import pandas as pd


from environment import Environment
from agent import Agent, ControlT, ZetaTensorT, HomeostaticT, ControlT, Zeta
from actions import Actions
from nets import Net_J, Net_f


class Algorithm:
    def __init__(self, difficulty: Difficulty, env: Environment, agent: Agent, actions: Actions, net_J: Net_J, net_f: Net_f):

        # ALGOS METAPARAMETERS ##################################

        # Iterations
        self.N_iter = 100

        # RL learning
        self.eps = 0.3  # random actions
        self.gamma = 0.99  # discounted rate
        self.tau = 0.001  # not used yet (linked with the target function)

        # Gradient descent
        self.learning_rate = 0.001
        self.asym_coeff = 100

        # plotting
        self.N_print = 1
        self.cycle_plot = self.N_iter - 1
        self.N_rolling = 5
        # save neural networks weights every N_save_weights step
        self.N_save_weights = 1000

        # CLASSES #########################################
        self.env = env
        self.agent = agent
        self.actions = actions
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

    def evaluate_action(self, action: int):
        """Return the score associated with the action.

        In this function, we do not seek to update the Net_F and Net_J,
         so we use the eval mode.
        But we still seek to use the derivative of the Net_F according to zeta.
         So we use require_grad = True.
        Generally, only the parameters of a neural network are on 
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
            [_zeta_tensor, torch.zeros(self.actions.nb_actions)])
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
        new_zeta_tensor = _zeta_tensor + self.agent.time_step * f
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
            if _zeta[i] + self.agent.x_star[i] < self.actions.min_resource:
                _zeta[i] = -self.agent.x_star[i] + self.actions.min_resource

        possible_actions = self.actions.actions_possible(self.env, self.agent)
        indexes_possible_actions = [i for i in range(
            self.actions.nb_actions) if possible_actions[i]]

        # The default action is doing nothing. Like people in real life.
        action = self.actions.inv_meaning_actions["not doing anything"]

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
            [_zeta, torch.zeros(self.actions.nb_actions)])
        zeta_u[len(_zeta) + action] = 1

        _new_zeta = self.actions.new_state(self.env, self.agent, action)  # actual choosen new_zeta
        
        if self.actions.meaning_actions[action] in ["walking_right", "walking_up", "walking_down", "walking_left"]:
            self.agent.zeta.last_direction = self.actions.meaning_actions[action]
        

        coeff = self.asym_coeff
        # set of big actions leading directly to the resources and 4 is for sleeping
        indices_going_directly = [
            self.actions.inv_meaning_actions[f"going direcly to resource {i}"]
            for i in range(self.difficulty.n_resources)]
        if action in [self.actions.inv_meaning_actions["sleeping"]] + indices_going_directly:
            coeff = 1

        predicted_new_zeta = _zeta + self.agent.time_step * \
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
            print("Action:", action, self.actions.meaning_actions[action])
            print("zeta:", _zeta)
            print("")

        if (k % self.N_save_weights) == 0:
            torch.save(self.net_J.state_dict(), 'weights/weights_net_J')
            torch.save(self.net_f.state_dict(), 'weights/weights_net_f')

        loss = np.array([Loss_f.detach().numpy(), Loss_J.detach().numpy()[0]])
        return action, loss

    def compute_mask(self, scale: int):
        """Compute the mask indicating if a discretized value is inside
        or outside the environment.

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
        """Plot the resource historic as a function of time.

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
                f'{last_action}: {self.actions.meaning_actions[last_action]} '),
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

        torch.save(self.net_J.state_dict(), 'weights/weights_net_J')
        torch.save(self.net_f.state_dict(), 'weights/weights_net_f')
