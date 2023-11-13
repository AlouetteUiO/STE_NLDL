import numpy as np
import pytest

from ste_NLDL.classes.config import Config
from ste_NLDL.classes.environment import POMDP as Environment
from ste_NLDL.classes.observationmodel import VergassolaPlumeModel as ObservationModel
from ste_NLDL.classes.bayes import AnalyticalBayes as Bayes

"""
Unit tests for Bayes
"""


def setup_1D():
    cfg = Config("test_ste/evaluate_infotaxis_1D_config.yml")
    mod = ObservationModel(cfg)
    bay = Bayes(cfg, mod)
    env = Environment(cfg, mod, bay)
    return mod, bay, env


def setup_2D():
    cfg = Config("test_ste/evaluate_infotaxis_2D_config.yml")
    mod = ObservationModel(cfg)
    bay = Bayes(cfg, mod)
    env = Environment(cfg, mod, bay)
    return mod, bay, env

def test_get_likelihood_1D():
    """
    Given Bayes' equation:

    p(theta|data) = p(data|theta) * p(theta) / p(data)
    or
    posterior     = likelihood    * prior    / evidence

    In our case: theta and data are discrete.
    Since a conditional probability distribution is a probability
    distribution over one or more variables given some evidence, 
    we know that:

    SUM_data p(data|theta) = 1      -> here the evidence is theta
    and
    SUM_theta p(theta|data) = 1     -> here the evidence is data
    
    p(data|theta) is given by hit_probabilities_meshgrid.
    The sum of the possible hits (h=0, h=1, h=2, h=3, etc.) should sum up
    to 1 at each of the possible source locations (theta).

    We can do:
    posterior ~ likelihood * prior
    and then normalize the posterior since we know that the sum over all
    possible source locations should be 1. 
    """
    mod, bay, env = setup_1D()
    assert bay.mod.Nhits == 5
    assert env.Ndim == 1
    assert env.Ngrid == 3
    assert env.Nflux == 1
    agent = np.array([2])
    hit_probabilities_meshgrid = mod._get_hit_probabilities_meshgrid()
    likelihood = bay._get_likelihood(hit_probabilities_meshgrid, agent)
    assert likelihood.shape == (bay.mod.Nhits, env.Nflux, bay.Ngrid)
    assert likelihood[0, 0, agent] == likelihood[1, 0, agent] == likelihood[2, 0, agent] == likelihood[3, 0, agent] == 0.0
    assert likelihood[4, 0, agent] == 1.0  # for Nhits-1
    # sum of probabilities for different hits should add up to 1.0 for each location in the grid
    assert (
        np.sum(likelihood[:, 0, 0])
        == np.sum(likelihood[:, 0, 1])
        == np.sum(likelihood[:, 0, 2])
        == 1.0
    )


def test_get_likelihood_2D():
    mod, bay, env = setup_2D()
    assert bay.mod.Nhits == 3
    assert env.Ndim == 2
    assert env.Ngrid == 5
    assert env.Nflux == 1
    agent = np.array([1, 2])
    hit_probabilities_meshgrid = mod._get_hit_probabilities_meshgrid()
    likelihood = bay._get_likelihood(hit_probabilities_meshgrid, agent)
    assert likelihood.shape == (bay.mod.Nhits, env.Nflux, bay.Ngrid, bay.Ngrid)
    assert (
        likelihood[0, 0, agent[0], agent[1]]
        == likelihood[1, 0, agent[0], agent[1]]
        == 0.0
    )
    assert likelihood[2, 0, agent[0], agent[1]] == 1.0  # for Nhits-1
    # sum of probabilities for different hits should add up to 1.0 for each location in the grid
    assert np.all(np.sum(likelihood, axis=0) == 1.0) 


def test_inference_iteration_1D():
    mod, bay, env = setup_1D()
    bay.hit_probabilities_meshgrid = mod._get_hit_probabilities_meshgrid()
    agent = np.array([1])
    prior = np.ones((env.Nflux, env.Ngrid), dtype=np.float32) / (env.Nflux * env.Ngrid)
    # with observation
    posterior, entropy, prob = bay._inference_iteration(
        prior, agent, observation=0
    )
    assert posterior.shape == prior.shape == (env.Nflux, bay.Ngrid)
    assert prob == None
    # no observation
    posterior, entropy, prob = bay._inference_iteration(
        prior, agent, observation=None
    )
    assert posterior.shape == (bay.mod.Nhits, env.Nflux, bay.Ngrid)
    assert prob.shape == (bay.mod.Nhits,)


def test_inference_iteration_2D():
    mod, bay, env = setup_2D()
    bay.hit_probabilities_meshgrid = mod._get_hit_probabilities_meshgrid()
    agent = np.array([1, 2])
    prior = np.ones((env.Nflux, env.Ngrid, env.Ngrid), dtype=np.float32) / (env.Nflux * env.Ngrid * env.Ngrid)
    # with observation
    posterior, entropy, prob = bay._inference_iteration(
        prior, agent, observation=0
    )
    assert posterior.shape == prior.shape == (env.Nflux, bay.Ngrid, bay.Ngrid)
    assert prob == None
    # no observation
    posterior, entropy, prob = bay._inference_iteration(
        prior, agent, observation=None
    )
    assert posterior.shape == (bay.mod.Nhits, env.Nflux, bay.Ngrid, bay.Ngrid)
    assert prob.shape == (bay.mod.Nhits,)
