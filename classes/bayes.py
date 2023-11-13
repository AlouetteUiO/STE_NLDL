import numpy as np
from ste_NLDL.functions.tools import get_entropy
import pytest


class Bayes:
    def __init__(self, cfg):
        self.index = cfg.get("methods/bayes")

        if self.index == 0:
            self.name = "Analytical Bayes"

    def inference_iteration(self, prior, agent, observation=None):
        """
        Perform one inference iteration
        """
        posterior, entropy, prob = self._inference_iteration(prior, agent, observation)
        return posterior, entropy, prob


class AnalyticalBayes(Bayes):
    """
    Analytical Bayes implementation (Bayes' theorem)
    this is a child class from Bayes
    """

    def __init__(self, cfg, mod):
        self.mod = mod

        # TODO These are set in multiple classes. Deal with this.
        self.Ndim = int(cfg.get("domain/Ndim"))
        self.Ngrid = int(cfg.get("domain/Ngrid"))

        self.flux_range = np.array(cfg.get("domain/flux_range"))
        self.Nflux = len(self.flux_range)

        super().__init__(cfg)

    def _get_likelihood(self, hit_probabilities_meshgrid, agent):
        """
        Extract the likelihood from hit_probabilities_meshgrid.

        Parameters
        ----------
        hit_probabilities_meshgrid: array of shape tuple([Nhits] + [Nflux] + [size] * Ndim)
            with size equal to 1 + 2 * Ngrid
            meshgrid of hit probabilities for all possible agent, source and flux combinations.
        agent: array of size Ndim
            agent location

        Returns
        -------
        likelihood: array of shape tuple([Nhits] + [Nflux] + [Ngrid] * Ndim)
            hit probabilities for all possible source and flux combinations, given the agent location.
        """
        index = np.array([self.Ngrid] * self.Ndim) - agent
        if self.Ndim == 1:
            likelihood = hit_probabilities_meshgrid[
                ..., index[0] : index[0] + self.Ngrid
            ]
        elif self.Ndim == 2:
            likelihood = hit_probabilities_meshgrid[
                ..., index[0] : index[0] + self.Ngrid, index[1] : index[1] + self.Ngrid
            ]
        elif self.Ndim == 3:
            likelihood = hit_probabilities_meshgrid[
                ...,
                index[0] : index[0] + self.Ngrid,
                index[1] : index[1] + self.Ngrid,
                index[2] : index[2] + self.Ngrid,
            ]
        return likelihood

    def _inference_iteration(self, prior, agent, observation=None):
        """
        Perform one inference iteration using Bayes' theorem.
        Either an observation is given and the function returns the posterior and entropy.
        Or there is no observation given and the function returns the expected posterior and expected entropy.

        1. The likelihood is extracted from hit_probabilities_meshgrid

        if observation:
            2. Posterior = prior * likelihood[observation]
            3. Normalize the posterior
            4. Compute entropy of the posterior

        if no observation:
            2. Expected posterior = prior * likelihood
            3. Hit probability is sum of the expected posterior
            4. Expected posterior is normalized by dividing by the hit probability
            5. Compute entropy of the expected posterior

        Parameters
        ----------
        prior: array of shape tuple([Nflux] + [Ngrid] * Ndim)
            current belief of the agent
        agent: array of size Ndim
            agent location
        observation: int or None
            number of hits detected by the agent

        Returns
        -------
        posterior: array of shape tuple([Nflux] + [Ngrid] * Ndim) (if observation) or
            tuple([Nhits] + [Nflux] + [Ngrid] * Ndim) (if no observation)
            updated belief of the agent
        entropy: float (if observation) or array of size Nhits (if no observation)
            entropy of the posterior
        prob: None (if observation) or array of size Nhits (if no observation)
            probability to reach updated belief state s' from the current
            belief state s for all possible observations given the agent's location
            (that followed from taking action a, so Pr(s'|s,a)).
        """

        likelihood = self._get_likelihood(self.mod.hit_probabilities_meshgrid, agent)
        # print(f"--- likelihood = \n{likelihood}")
        # print(f"--- prior = \n{prior}")

        assert np.sum(prior) == pytest.approx(1.0)

        if observation != None:
            # Compute posterior
            posterior = np.multiply(prior, likelihood[observation])  
            # print(f"--- posterior = \n{posterior}")
            posterior /= np.sum(posterior)
            # print(f"--- posterior after normalizing = \n{posterior}")

            # Compute entropy
            entropy = get_entropy(probability_distributon=posterior)

            assert entropy != -np.inf
            assert np.sum(posterior) == pytest.approx(1.0)
            assert posterior.shape == tuple([self.Nflux] + [self.Ngrid] * self.Ndim)
            return posterior, entropy, None

        else:  # observation == None
            # Compute expected posterior
            posterior = np.multiply(prior, likelihood)
            # print(f"--- posterior = prior * likelihood =\n{posterior}")
            prob = np.sum(posterior, axis=tuple(range(1, posterior.ndim)))
            # print(f"--- prop = {prob}")

            # normalizing
            newshape = tuple([self.mod.Nhits] + [1] * (self.Ndim + 1))
            posterior /= np.reshape(prob, newshape)
            # print(f"--- posterior / prob = \n{posterior}")

            # Compute expected entropy
            entropy = get_entropy(
                probability_distributon=posterior, axis=tuple(range(1, posterior.ndim))
            )
            # print(f"--- entropy = \n{entropy}")

            assert np.sum(prob) == pytest.approx(1.0)
            assert np.sum(posterior) == pytest.approx(self.mod.Nhits)
            assert posterior.shape == tuple(
                [self.mod.Nhits] + [self.Nflux] + [self.Ngrid] * self.Ndim
            )
            assert entropy.shape == (self.mod.Nhits,)
            assert prob.shape == (self.mod.Nhits,)
            return posterior, entropy, prob
