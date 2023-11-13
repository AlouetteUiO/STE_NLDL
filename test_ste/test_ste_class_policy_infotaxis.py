import pytest
import numpy as np

from ste_NLDL.classes.config import Config
from ste_NLDL.classes.environment import POMDP as Environment
from ste_NLDL.classes.policy import Infotaxis as Policy
from ste_NLDL.classes.observationmodel import VergassolaPlumeModel as ObservationModel
from ste_NLDL.classes.bayes import AnalyticalBayes as Bayes
from ste_NLDL.functions.tools import get_entropy

"""
Unit tests for infotaxis development
These tests give the same results (prior, likelihood (p_evidence in Otto), expected posterior, entropy) as Otto
"""

cfg = Config("test_ste/evaluate_infotaxis_2D_config.yml")
mod = ObservationModel(cfg)
bay = Bayes(cfg, mod)
env = Environment(cfg, mod, bay)
pol = Policy(cfg, env, bay)


def setup_1():
    env.reset()
    env.belief *= 0  # overwrite belief
    env.belief += 1 / env.Ngrid**env.Ndim
    env.belief[0, env.Ngrid // 2 + 1, env.Ngrid // 2] = 0.2
    env.belief[0, env.Ngrid // 2, env.Ngrid // 2 + 1] = 0.8
    env.belief /= np.sum(env.belief)
    return


def setup_2():
    env.reset()
    env.belief *= 0  # overwrite belief
    env.belief[0, env.Ngrid // 2 + 1, env.Ngrid // 2] = 0.5
    env.belief[0, env.Ngrid // 2, env.Ngrid // 2 + 1] = 0.5
    env.belief /= np.sum(env.belief)
    return


def test_behaviour_infotaxis():
    setup_1()
    # env.belief =
    #  [[[0.02083333 0.02083333 0.02083333 0.02083333 0.02083333]
    #    [0.02083333 0.02083333 0.02083333 0.02083333 0.02083333]
    #    [0.02083333 0.02083333 0.02083333 0.41666667 0.02083333]
    #    [0.02083333 0.02083333 0.10416667 0.02083333 0.02083333]
    #    [0.02083333 0.02083333 0.02083333 0.02083333 0.02083333]]]
    # and agent in center
    action = pol.select_best_action()
    assert action == 3  # go to right.
    next_agent, _ = env.move(env.agent, action)
    assert np.all(next_agent == np.array([2, 3]))
    setup_2()
    # env.belief =
    #  [[[0.  0.  0.  0.  0. ]
    #    [0.  0.  0.  0.  0. ]
    #    [0.  0.  0.  0.5 0. ]
    #    [0.  0.  0.5 0.  0. ]
    #    [0.  0.  0.  0.  0. ]]]
    action = pol.select_best_action()
    assert action == 1 or action == 3  # go down or to the right
