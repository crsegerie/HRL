import torch
import unittest

from agent import Agent
from environment import Environment
from algorithm import Algorithm
from nets import Net_J, Net_f
from utils import set_all_seeds
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# class TestEnvironnement(unittest.TestCase):

#     def test_upper(self):
#         self.assertEqual('foo'.upper(), 'FOO')




class TestHRL(unittest.TestCase):

    def test_end_result(self):
        seed = 0
        set_all_seeds(seed)

        env = Environment()
        agent = Agent()
        net_J = Net_J()
        net_f = Net_f()
        algo = Algorithm(env, agent, net_J, net_f)

        algo.simulation()
        _zeta_tensor = algo.agent.zeta._zeta_tensor
        wanted_results = torch.tensor(
            [-0.9050, -1.9050, -2.9050, -3.9050,  0.1154,  0.1083,  4.7972,  1.0833, 3.0000])

        bool = torch.allclose(_zeta_tensor, wanted_results, rtol=1e-03)

        self.assertEqual(bool, True)


if __name__ == '__main__':
    unittest.main()
