"""Homeostatic reinforcement learning.

Described in the paper:
Continuous Homeostatic Reinforcement Learning
for Self-Regulated Autonomous Agents.
Authors : Hugo Laurençon, Charbel-Raphaël Ségerie,
Johann Lussange, Boris S. Gutkin.
"""
from typing import Literal
from agent import Agent
from environment import Environment
from algorithm import Algorithm
from nets import Net_J, Net_f
from utils import set_all_seeds, Difficulty
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def main():
    # Random seed
    seed = 0
    set_all_seeds(seed)
    
    difficulty = Difficulty(level="MEDIUM")

    env = Environment(difficulty)
    agent = Agent(difficulty)
    net_J = Net_J()
    net_f = Net_f()
    algo = Algorithm(difficulty, env, agent, net_J, net_f)


    algo.simulation()



if __name__ == "__main__":
    main()