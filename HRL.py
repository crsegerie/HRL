"""Homeostatic reinforcement learning.

Described in the paper:
Continuous Homeostatic Reinforcement Learning
for Self-Regulated Autonomous Agents.
"""
from agent import Agent
from environment import Environment
from algorithm import Algorithm
from nets import Net_J, Net_f
from utils import set_all_seeds
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# Random seed
seed = 0
set_all_seeds(seed)


env = Environment()
agent = Agent()
net_J = Net_J()
net_f = Net_f()
algo = Algorithm(env, agent, net_J, net_f)


algo.simulation()
