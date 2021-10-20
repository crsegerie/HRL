from typing import Any, Dict

from environment import Environment
from agent import ZetaTensorT, Zeta, Agent
from utils import Hyperparam

import torch
import numpy as np

from typing import Callable

import pandas as pd


class Actions:
    """Defining the possible actions for the agent
    in its environment.
    """

    def __init__(self, hyperparam: Hyperparam) -> None:

        self.hp = hyperparam

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
                    [0.]*self.hp.difficulty.n_resources + [0.01, 0., self.hp.cst_agent.walking_speed, 0.])
            elif direction == 'left':
                control_walking = torch.tensor(
                    [0.]*self.hp.difficulty.n_resources + [0.01, 0., -self.hp.cst_agent.walking_speed, 0.])
            elif direction == 'up':
                control_walking = torch.tensor(
                    [0.]*self.hp.difficulty.n_resources + [0.01, 0., 0., self.hp.cst_agent.walking_speed])
            elif direction == 'down':
                control_walking = torch.tensor(
                    [0.]*self.hp.difficulty.n_resources + [0.01, 0., 0., -self.hp.cst_agent.walking_speed])

            def new_state_walking(agent: Agent, env: Environment) -> Zeta:
                new_zeta = Zeta(self.hp)
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
            control_sleeping = torch.tensor([0.]*self.hp.difficulty.n_resources + [0., -0.001, 0., 0.])
            duration_sleep = self.n_min_time_sleep * self.hp.cst_algo.time_step
            new_zeta = Zeta(self.hp)
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
            new_zeta = Zeta(self.hp)
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
            if (res < 0) or (res >= self.hp.difficulty.n_resources):
                raise ValueError('res should be between 0 and n_resources-1.')
            else:
                def new_state_consuming_resource(agent: Agent, env: Environment) -> Zeta:
                    control_consuming_resource = [0.]*agent.zeta.shape
                    control_consuming_resource[res] = 0.1
                    control_consuming_resource = torch.tensor(control_consuming_resource)
                    new_zeta = Zeta(self.hp)
                    new_zeta.tensor = agent.euler_method(agent.zeta, control_consuming_resource)
                    return new_zeta

                def constraints_consuming_resource(agent: Agent, env: Environment) -> bool:
                    return ((agent.zeta.aware_energy < self.max_tired)
                            and env.is_near_resource(agent.zeta.x, agent.zeta.y, res)
                            and (agent.zeta.resource(res) < 8))

                return new_state_consuming_resource, constraints_consuming_resource
        
        # ACTIONS OF CONSUMING A RESOURCE

        for res in range(self.hp.difficulty.n_resources):
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
            if (res < 0) or (res >= self.hp.difficulty.n_resources):
                raise ValueError('res should be between 0 and n_resources-1.')
            else:
                def new_state_going_to_resource(agent: Agent, env: Environment) -> Zeta:
                    control_going_to_resource = torch.tensor(
                        [0.]*self.hp.difficulty.n_resources + [0.01, 0., 0., 0.])
                    dist = env.distance_to_resource(agent.zeta.x, agent.zeta.y, res)
                    duration_walking = dist * self.hp.cst_algo.time_step / self.hp.cst_agent.walking_speed

                    new_zeta = Zeta(self.hp)
                    new_zeta.tensor = agent.integrate_multiple_steps(
                        duration_walking, agent.zeta, control_going_to_resource)
                    new_zeta.x = self.hp.cst_env.resources[res].x
                    new_zeta.y = self.hp.cst_env.resources[res].y

                    return new_zeta

                def constraints_going_to_resource(agent: Agent, env: Environment) -> bool:
                    new_zeta = new_state_going_to_resource(agent, env)
                    return ((new_zeta.aware_energy < self.max_tired)  
                            and (new_zeta.muscular_fatigue < 6)
                            and env.is_resource_visible(agent.zeta.x, agent.zeta.y, res)
                            and (env.distance_to_resource(agent.zeta.x, agent.zeta.y, res) > 0))

                return new_state_going_to_resource, constraints_going_to_resource

        # ACTIONS OF GOING TO A RESOURCE

        for res in range(self.hp.difficulty.n_resources):
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

        self.df = pd.DataFrame(actions_list)

        self.n_actions = len(self.df)



