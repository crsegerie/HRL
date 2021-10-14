from typing import Any, Dict

from environment import Environment
from agent import ZetaTensorT, Zeta, Agent
from utils import Difficulty

import torch
import numpy as np

from typing import Callable

import pandas as pd


class Actions:
    """Defining the possible actions for the agent
    in its environment.
    """

    def __init__(self, difficulty: Difficulty, agent: Agent) -> None:
        self.difficulty = difficulty

        # ACTIONS METAPARAMETERS ##################################

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
            "walking_right": [0.]*n_resources + [0.01, 0., agent.walking_speed, 0.],
            "walking_left": [0.]*n_resources + [0.01, 0., -agent.walking_speed, 0.],
            "walking_up": [0.]*n_resources + [0.01, 0., 0., agent.walking_speed],
            "walking_down": [0.]*n_resources + [0.01, 0., 0., -agent.walking_speed],
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
        self.n_actions = len(self.actions_controls) + n_resources

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

    def actions_possible(self, env: Environment, agent: Agent):
        """Return a list of bool showing which action is permitted or not.

        + 4 for the action of going to a resource after seeing it
        """
        zeta = agent.zeta

        # TODO:: boucle sur pandas
        # There are 4 more possible actions:
        # The 4 last actions correspond to going direcly to each of the
        # 4 ressources if possible.
        possible_actions = [True for i in range(
            len(self.actions_controls) + agent.zeta.shape)]

        # If the agent is too tired in his muscle
        if zeta.muscular_fatigue >= self.constraints[0]:
            # walking is no more possible
            possible_actions[self.inv_meaning_actions["walking_right"]] = False
            possible_actions[self.inv_meaning_actions["walking_left"]] = False
            possible_actions[self.inv_meaning_actions["walking_up"]] = False
            possible_actions[self.inv_meaning_actions["walking_down"]] = False

        # He cannot escape the environment when walking.
        # TODO: zeta_meaning["x"] = 6
        # TODO: x_indice = 6
        str_walkings = ["walking_up", "walking_down",
                        "walking_right", "walking_left"]
        for str_walking in str_walkings:
            x_walk = zeta.x + agent.time_step * \
                self.actions_controls[str_walking][agent.zeta.x_indice]
            y_walk = zeta.y + agent.time_step * \
                self.actions_controls[str_walking][agent.zeta.y_indice]
            x_walk, y_walk = float(x_walk), float(y_walk)
            if not env.is_point_inside(x_walk, y_walk):
                possible_actions[self.inv_meaning_actions[str_walking]] = False

        # The agent cannot sleep if he is not enouth tired.
        if zeta.aware_energy <= self.constraints[self.inv_meaning_actions["sleeping"]]:
            possible_actions[self.inv_meaning_actions["sleeping"]] = False

        # If the agent is too sleepy, the only possible action is to sleep.
        if zeta.aware_energy >= self.max_tired:
            possible_actions = [False for p in possible_actions]
            possible_actions[self.inv_meaning_actions["sleeping"]] = True

        def is_near_resource(resource_i: int):
            dist = (zeta.x - env.resources[resource_i].x)**2 + (
                zeta.y - env.resources[resource_i].y)**2
            radius = env.resources[resource_i].r**2
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

        n_resources = agent.zeta.difficulty.n_resources
        for resource in range(n_resources):
            check_resource_eatable(resource)

        def is_resource_visible(resource: int):
            """Check if segment between agent and resource i is visible"""
            xa = float(zeta.x)
            xb = env.resources[resource].x
            ya = float(zeta.y)
            yb = env.resources[resource].y
            return env.is_segment_inside(xa, xb, ya, yb)

        # big actions : seing a resource and eat it.
        for resource in range(n_resources):
            if not is_resource_visible(resource):
                possible_actions[len(possible_actions) -
                                 n_resources + resource] = False

        return possible_actions

    def going_and_get_resource(self, env: Environment, agent: Agent, resource_i: int) -> ZetaTensorT:
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
        new_zeta_tensor = agent.zeta.tensor

        agent_x = agent.zeta.x
        agent_y = agent.zeta.y
        resource_x = env.resources[resource_i].x
        resource_y = env.resources[resource_i].y
        distance = np.sqrt((agent_x - resource_x)**2 +
                           (agent_y - resource_y)**2)

        if distance == 0:
            # If the agent is already on the resource, then consuming it is done instantly
            u = self.actions_controls["not doing anything"]

            agent.zeta.tensor = agent.euler_method(agent.zeta, u)

            return new_zeta_tensor

        # If the agent is at a distance d from the resource,
        # it will first need to walk
        # to consume it. Thus, we integrate the differential
        # equation of its internal state during this time
        time_to_walk = distance * agent.time_step / agent.walking_speed

        # We take only homeostatic part of the control
        # So the direction does not matter here
        # TODO: LI norm
        control = self.actions_controls["walking_up"][:agent.zeta.n_homeostatic]
        new_zeta_tensor = agent.integrate_multiple_steps(
            time_to_walk, agent.zeta, control)

        new_zeta_tensor[agent.zeta.x_indice] = env.resources[resource_i].x
        new_zeta_tensor[agent.zeta.y_indice] = env.resources[resource_i].y
        return new_zeta_tensor

    def new_state(self, env: Environment, agent: Agent, a: int) -> ZetaTensorT:
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
            duration_sleep = self.n_min_time_sleep * agent.time_step
            control_sleep = self.actions_controls["sleeping"][:agent.zeta.n_homeostatic]
            new_zeta.tensor = agent.integrate_multiple_steps(
                duration_sleep, agent.zeta, control_sleep)

        # going direcly to resource
        elif a in [self.inv_meaning_actions[f"going direcly to resource {i}"]
                   for i in range(self.difficulty.n_resources)]:
            resource_i = int(self.meaning_actions[a][-1:])
            self.going_and_get_resource(env, agent, resource_i)

        # Other actions: elementary actions
        else:
            u = torch.zeros(agent.zeta.shape)
            a_meaning = list(self.actions_controls.keys())[a]
            u = self.actions_controls[a_meaning]

            # Euler method to calculate the new zeta.
            new_zeta.tensor = agent.euler_method(agent.zeta, u)

        return new_zeta.tensor












    def init2(self, difficulty: Difficulty, agent: Agent) -> None:

        self.difficulty = difficulty

        # Minimum duration for the action of sleeping
        self.n_min_time_sleep = 1000
        # If the agent is too tired, it automatically sleeps.
        self.max_tired = 10

        # If any of the agent's resources are less than min_resource,
        # we raise it to min_resource.
        # Because the dynamics is controlled by an exponential function,
        # if one resource equals 0, it can never be raised again.
        self.min_resource = 0.1

        actions_list = []

        # USEFUL FOR ACTIONS OF WALKING

        def new_state_and_constraints_walking(direction: str):
            if direction not in ['right', 'left', 'up', 'down']:
                raise ValueError('direction should be right, left, up or down.')
            elif direction == 'right':
                control_walking = torch.tensor(
                    [0.]*difficulty.n_resources + [0.01, 0., agent.walking_speed, 0.])
            elif direction == 'left':
                control_walking = torch.tensor(
                    [0.]*difficulty.n_resources + [0.01, 0., -agent.walking_speed, 0.])
            elif direction == 'up':
                control_walking = torch.tensor(
                    [0.]*difficulty.n_resources + [0.01, 0., 0., agent.walking_speed])
            elif direction == 'down':
                control_walking = torch.tensor(
                    [0.]*difficulty.n_resources + [0.01, 0., 0., -agent.walking_speed])

            def new_state_walking(agent: Agent, env: Environment) -> Zeta:
                new_zeta = Zeta(difficulty)
                new_zeta.tensor = agent.euler_method(agent.zeta, control_walking)
                return new_zeta

            def constraints_walking(agent: Agent, env: Environment) -> bool:
                new_zeta = new_state_walking(agent, env)
                return ((agent.zeta.aware_energy < self.max_tired) 
                        and (agent.zeta.muscular_fatigue < 6) 
                        and env.is_point_inside(new_zeta.x, new_zeta.y))

            return new_state_walking, constraints_walking

        # ACTIONS OF WALKING RIGHT, LEFT, UP AND DOWN

        for direction in ['right', 'left', 'up', 'down']:
            new_state_walking, constraints_walking = new_state_and_constraints_walking(direction)
            action_walking = {
                "name": f"walking_{direction}",
                "definition": f"Walking one step {direction}.",
                "new_state": new_state_walking,
                "constraints": constraints_walking,
                "coefficient_loss": 1,
            }
            actions_list.append(action_walking)

        # ACTION OF SLEEPING

        def new_state_sleeping(agent: Agent, env: Environment) -> Zeta:
            control_sleeping = torch.tensor([0.]*difficulty.n_resources + [0., -0.001, 0., 0.])
            duration_sleep = self.n_min_time_sleep * agent.time_step
            new_zeta = Zeta(difficulty)
            new_zeta.tensor = agent.integrate_multiple_steps(
                duration_sleep, agent.zeta, control_sleeping)
            return new_zeta

        def constraints_sleeping(agent: Agent, env: Environment) -> bool:
            return (agent.zeta.aware_energy > 1)

        action_sleeping = {
            "name": "sleeping",
            "definition": "Sleeping for a fixed time period to recover from muscular and aware tiredness.",
            "new_state": new_state_sleeping,
            "constraints": constraints_sleeping,
            "coefficient_loss": 100,
        }
        actions_list.append(action_sleeping)

        # ACTION OF DOING NOTHING

        def new_state_doing_nothing(agent: Agent, env: Environment) -> Zeta:
            control_doing_nothing = torch.tensor([0.]*agent.zeta.shape)
            new_zeta = Zeta(difficulty)
            new_zeta.tensor = agent.euler_method(agent.zeta, control_doing_nothing)
            return new_zeta

        def constraints_doing_nothing(agent: Agent, env: Environment) -> bool:
            return (agent.zeta.aware_energy < self.max_tired)

        action_doing_nothing = {
            "name": "doing_nothing",
            "definition": "Standing still and doing nothing.",
            "new_state": new_state_doing_nothing,
            "constraints": constraints_doing_nothing,
            "coefficient_loss": 1,
        }
        actions_list.append(action_doing_nothing)

        # USEFUL FOR ACTIONS OF CONSUMING A RESOURCE

        def new_state_and_constraints_consuming_resource(res: int):
            if (res < 0) or (res >= difficulty.n_resources):
                raise ValueError('res should be between 0 and n_resources-1.')
            else:
                def new_state_consuming_resource(agent: Agent, env: Environment) -> Zeta:
                    control_consuming_resource = [0.]*agent.zeta.shape
                    control_consuming_resource[res] = 0.1
                    control_consuming_resource = torch.tensor(control_consuming_resource)
                    new_zeta = Zeta(difficulty)
                    new_zeta.tensor = agent.euler_method(agent.zeta, control_consuming_resource)
                    return new_zeta

                def constraints_consuming_resource(agent: Agent, env: Environment) -> bool:
                    return ((agent.zeta.aware_energy < self.max_tired)
                            and env.is_near_resource(agent.zeta.x, agent.zeta.y, res)
                            and (agent.zeta.resource(res) < 8))

                return new_state_consuming_resource, constraints_consuming_resource
        
        # ACTIONS OF CONSUMING A RESOURCE

        for res in range(difficulty.n_resources):
            new_state_consuming_resource, constraints_consuming_resource = new_state_and_constraints_consuming_resource(res)
            action_consuming_resource = {
                "name": f"consuming_resource_{res}",
                "definition": f"Consuming resource {res}.",
                "new_state": new_state_consuming_resource,
                "constraints": constraints_consuming_resource,
                "coefficient_loss": 1,
            }
            actions_list.append(action_consuming_resource)

        # USEFUL FOR ACTIONS OF GOING TO A RESOURCE

        def new_state_and_constraints_going_to_resource(res: int):
            if (res < 0) or (res >= difficulty.n_resources):
                raise ValueError('res should be between 0 and n_resources-1.')
            else:
                def new_state_going_to_resource(agent: Agent, env: Environment) -> Zeta:
                    control_going_to_resource = torch.tensor(
                        [0.]*difficulty.n_resources + [0.01, 0., 0., 0.])
                    dist = env.distance_to_resource(agent.zeta.x, agent.zeta.y, res)
                    duration_walking = dist * agent.time_step / agent.walking_speed

                    new_zeta = Zeta(difficulty)
                    new_zeta.tensor = agent.integrate_multiple_steps(
                        duration_walking, agent.zeta, control_going_to_resource)
                    new_zeta.tensor.x = env.resources[res].x
                    new_zeta.tensor.y = env.resources[res].y

                    return new_zeta

                def constraints_going_to_resource(agent: Agent, env: Environment) -> bool:
                    new_zeta = new_state_going_to_resource(agent, env)
                    return ((new_zeta.aware_energy < self.max_tired)  
                            and (new_zeta.muscular_fatigue < 6)
                            and env.is_resource_visible(agent.zeta.x, agent.zeta.y, res)
                            and (env.distance_to_resource(agent.zeta.x, agent.zeta.y, res) > 0))

                return new_state_going_to_resource, constraints_going_to_resource

        # ACTIONS OF GOING TO A RESOURCE

        for res in range(difficulty.n_resources):
            new_state_going_to_resource, constraints_going_to_resource = new_state_and_constraints_going_to_resource(res)
            action_going_to_resource = {
                "name": f"going_to_resource_{res}",
                "definition": f"Going to resource {res}.",
                "new_state": new_state_going_to_resource,
                "constraints": constraints_going_to_resource,
                "coefficient_loss": 100,
            }
            actions_list.append(action_going_to_resource)

        # CREATION OF THE DATAFRAME

        self.actions = pd.DataFrame(actions_list)

        self.n_actions = len(self.actions)



