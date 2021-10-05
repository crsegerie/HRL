import torch
import unittest

import numpy as np
import matplotlib.pyplot as plt

from agent import Agent
from environment import Environment
from algorithm import Algorithm
from nets import Net_J, Net_f
from utils import set_all_seeds, Difficulty
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class TestEnvironnement(unittest.TestCase):

    def test_is_point_inside_medium(self):
        difficulty = Difficulty(level="MEDIUM")
        env = Environment(difficulty)

        Points = [
            (1.1, 1.1, True),  # up right A
            (0.9, 0.9, False),  # down left A
            (0.9, 5, False),  # left B
            (1.1, 4.9, True),  # right down B
        ]

        for (x, y, inside) in Points:
            self.assertEqual(env.is_point_inside(x, y), inside)

    def test_is_point_inside_easy(self):
        """EASY : 10x10 square environneemnt."""
        difficulty = Difficulty(level="EASY")
        env = Environment(difficulty)

        Points = [
            (-1, 1, False),
            (1, 1, True),
            (9, 1, True),
            (11, 1, False),
        ]

        for (x, y, inside) in Points:
            self.assertEqual(env.is_point_inside(x, y), inside)

    def test_is_segment_inside_medium(self):
        difficulty = Difficulty(level="MEDIUM")

        env = Environment(difficulty)

        segments = [
            # xa, ya, xb, yb,
            (1.1, 1.1, 0.9, 0.9, False),  # up right A - down left A
            (0.9, 0.9, 1.1, 1.1, False),  # down left A - up right A
            (0.9, 0.9, 1.1, 4.9, False),  # down left A - right down B
            (1.1, 1.1, 1.1, 4.9, True),  # up right A  - right down B
            (1.1, 1.1, 1.1, 5.1, False),  # up right A  - right up B
            (1.5, 1.5, 1.5, 4.5, True), # other test
            (1.5, 4.5, 1.5, 1.5, True), # other test
            (1.5, 6.5, 1.5, 1.5, False), # other test
        ]

        for (xa, ya, xb, yb, inside) in segments:
            self.assertEqual(env.is_segment_inside(xa, xb, ya, yb), inside)

    def test_visualization_env(self):
        difficulty = Difficulty(level="MEDIUM")
        env = Environment(difficulty)

        values = np.zeros((90, 60))
        for i in range(90):  # x
            for j in range(60):  # y
                values[i, j] = 1*env.is_point_inside(i/10, j/10)

        plt.imshow(values.T, cmap='cool', interpolation='nearest')
        plt.gca().invert_yaxis()
        plt.show()

class TestHRL(unittest.TestCase):

    def test_end_result(self):
        seed = 0
        set_all_seeds(seed)

        difficulty = Difficulty(level="MEDIUM")
        env = Environment(difficulty)
        agent = Agent(difficulty)
        net_f = Net_f(shape_zeta=agent.zeta.shape, n_tot_actions=difficulty.n_actions)
        net_J = Net_J(shape_zeta=agent.zeta.shape)
        algo = Algorithm(difficulty, env, agent, net_J, net_f)

        algo.simulation()
        _zeta_tensor = algo.agent.zeta.tensor
        wanted_results = torch.tensor(
            [-0.9050, -1.9050, -2.9050, -3.9050,  0.1154,  0.1083,  4.7972,  1.0833, 3.0000])

        #bool = torch.allclose(_zeta_tensor, wanted_results, rtol=1e-03)

        #self.assertEqual(bool, True)


if __name__ == '__main__':
    unittest.main()


