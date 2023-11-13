import numpy as np
from scipy.special import kn
from scipy.stats import poisson
from pytest import approx
from itertools import product


class ObservationModel:
    """
    ObservationModel superclass
    """

    def __init__(self, cfg):
        self.index = cfg.get("methods/observationmodel")

        if self.index == 0:
            self.name = "gaussian plume model"
            raise Exception("Not implemented")
        elif self.index == 1:
            self.name = "LES"
            raise Exception("Not implemented")
        elif self.index == 2:
            self.name = "Vergassola plume model"

    def get_observation(self, agent, source, flux, flux_index, seed=None):
        """get an observation for a given agent location, source location and flux"""
        observation = self._get_observation(agent, source, flux, flux_index, seed)
        return observation

class VergassolaPlumeModel(ObservationModel):
    """
    VergassolaPlumeModel subclass
    This is a child class from ObservationModel

    The model comes from Vergassola (2007), and is used by Loisy and Eloy (2022)
    """

    def __init__(self, cfg):
        # Set mandatory domain variables
        # TODO This is also done in environment, has to be done differently
        self.Ngrid = int(cfg.get("domain/Ngrid"))
        self.Ndim = int(cfg.get("domain/Ndim"))
        self.V = float(cfg.get("domain/V")) # non-dimensional wind speed
        self.D = float(cfg.get("domain/D")) # non-dimensional diffusivity
        self.flux_range = list(cfg.get("domain/flux_range"))
        self.Nflux = len(self.flux_range)

        # Set fixed values
        self.tau = 10000000 # non-dimensional diffusion length
        self.dx_over_a = 2.0

        # Autoset lambda_over_dx
        self.lambda_over_dx = np.sqrt((self.D * self.tau) / (1 + ((self.V**2 * self.tau) / (4*self.D)) ))

        # Autoset Nhits
        self.Nhits = self._autoset_Nhits()

        # Get meshgrid of all possible hit probabilities
        self.hit_probabilities_meshgrid = self._get_hit_probabilities_meshgrid()

        super().__init__(cfg)

    def _get_hit_probabilities_meshgrid(self):
        """
        hit_probabilities_meshgrid is a meshgrid of all possible hit probabilities based
        on the location of the agent from the source, downwind location of the agent from the source
        and the flux. The agent is in the origin of hit_probabilities_meshgrid.

        Returns
        -------
        hit_probabilities_meshgrid: array of shape tuple([Nhits] + [Nflux] + [size] * Ndim)
            with size equal to 1 + 2 * self.Ngrid
            meshgrid of hit probabilities for all possible agent, source and flux combinations.
        """

        size = 1 + 2 * self.Ngrid
        origin = [self.Ngrid] * self.Ndim 
        # print(f"size = {size} and origin = {origin}")

        hit_probabilities_meshgrid = np.zeros(
            [self.Nhits] + [self.Nflux] + [size] * self.Ndim
        )

        indices = [range(1 + 2 * self.Ngrid) for _ in range(self.Ndim)]
        for i, flux in enumerate(self.flux_range):
            for index in product(*indices):
                hit_probabilities = self._get_hit_probabilities(
                    agent=origin, source=index, flux=flux
                )  
                if self.Ndim == 1:
                    hit_probabilities_meshgrid[:, i, index[0]] = hit_probabilities
                elif self.Ndim == 2:
                    hit_probabilities_meshgrid[:, i, index[0], index[1]] = hit_probabilities
                elif self.Ndim == 3:
                    hit_probabilities_meshgrid[:, i, index[0], index[1], index[2]] = hit_probabilities

        return hit_probabilities_meshgrid

    def _autoset_Nhits(self):
        """
        Set the number of possible hits the agent can receive. 
        This is the length of the range from 0 to hits_max of 
        the largest flux in flux_range. 

        Returns
        -------
        Nhits: int
            Number of possible hits the agent can receive
            If Nhits = 3 then the agent can receive hit = 0,
            hit = 1 or hit = 2.
        """
        Nhits = self._get_hits_max(flux = np.max(self.flux_range)) + 1
        print(f"autoset Nhits = {Nhits}")
        return Nhits
    
    def _get_hits_max(self, flux):
        """ 
        Set a maximum number of hits. We base the maximum number of hits on the average 
        number of hits in the grid cell downwind from the source. Vergassola mentions: 
        "Typical fluctuations [in the number of particles] are of the order of the square root of the mean".

        Arguments
        ---------
        flux: float, int or array
            dimensionless source intensity
            (R_dt in Loisy and Eloy code or I in Loisy and Eloy documentation)

        Returns
        -------
        hits_max: int
            Highest number of hits the agent can receive. 
        """
        mu_at_dx = self._get_mu(flux=flux, d_over_dx=1.0, d_over_dx_downwind=1.0)
        hits_max = np.ceil(mu_at_dx + np.sqrt(mu_at_dx))
        hits_max = hits_max.astype('int')
        # print(f"hits_max = {hits_max}")
        return hits_max

    def _get_mu_concentration(self, flux, d_over_dx, d_over_dx_downwind):
        """ 
        Compute the mean concentration at a domain location. 
        """

        if self.Ndim == 1:
            raise Exception

        elif self.Ndim == 2:
            mu_conc = (
                (flux / (2 * np.pi * self.D))
                * np.exp( (d_over_dx_downwind * self.V) / (2 * self.D) )
                * kn(0, d_over_dx / self.lambda_over_dx)
            )  

        elif self.Ndim == 3:
            mu_conc = (
                (flux / (4 * np.pi * self.D * d_over_dx))
                * np.exp( (d_over_dx_downwind * self.V) / (2 * self.D) )
                * np.exp( - d_over_dx / self.lambda_over_dx )
            )
        
        else:
            raise Exception

        return mu_conc

    def _get_mu(self, flux, d_over_dx, d_over_dx_downwind):
        """
        Compute the mean number of hits at a domain location.

        Arguments
        ---------
        flux: int
            dimensionless source intensity
            (R_dt in Loisy and Eloy code or I in Loisy and Eloy documentation)
        d_over_dx: float or int
            distance between agent and source expressed in number of grid cells
            d_over_dx = 0 is the source location
        d_over_dx_downwind: float or int
            distance between x position of agent and x position of source expressed in number of grid cells
            d_over_dx_downwind is negative upwind of the source and positive downwind from the source

        Returns
        -------
        mu: float
            mean number of hits for given flux and distance of agent from the source
        """

        if self.Ndim == 1:
            a_over_lambda = 1 / (self.dx_over_a * self.lambda_over_dx)
            mu = (flux * (1 / (1 - a_over_lambda)) * np.exp(-d_over_dx / self.lambda_over_dx))

        elif self.Ndim == 2:
            mu_conc = self._get_mu_concentration(flux, d_over_dx, d_over_dx_downwind)
            lambda_over_a = self.dx_over_a * self.lambda_over_dx
            mu = (2 * np.pi * self.D * mu_conc) / np.log(lambda_over_a) 
        
        elif self.Ndim == 3:
            mu_conc = self._get_mu_concentration(flux, d_over_dx, d_over_dx_downwind)
            a_over_dx = 1 / self.dx_over_a
            mu = 4 * np.pi * self.D * a_over_dx * mu_conc

        else:
            raise Exception

        # Set mu = inf for d_over_dx == 0 (at source location).
        if isinstance(d_over_dx, (float, int)) and d_over_dx == 0.0:
            mu = np.inf
        elif isinstance(d_over_dx, (list, tuple, np.ndarray)):
            filter = d_over_dx == 0.0
            mu[filter] = np.inf

        # Change inf to very large number
        mu = np.nan_to_num(mu)
        return mu

    def _Poisson(self, mu, hit):
        """
        Compute the probability of a hit given the mean number of hits
        at that location (following the Poisson distributon)
        """
        if np.any(hit < self.Nhits):
            probability = poisson.pmf(hit, mu)
        elif hit == self.Nhits:
            raise Warning(
                "We work with Nhits. The total probability at a node should sum up to one. \
                The probability for hit == Nhit is obtained by 1 - sum of probability for hit < Nhits."
            )
        else:
            raise Exception("Hit should not exceed Nhits")
        return probability

    def _get_hit_probabilities(self, agent, source, flux):
        """
        Compute the hit probabilities, working with Nhits
        1. Compute the dimensionless distance between source and agent.
        2. Compute the mean number of hits at location of the agent.
        3. Compute the hit probabilities with the Poisson distribution.

        Parameters 
        ----------
        agent: list, tuple or array of size Ndim
            agent location
        source: list, tuple or array of size Ndim
            true source location
        flux: int
            true source strength

        Returns
        -------
        hit_probabilities: array of size Nhits
            probability to observe number of hits based on the observation model.
            the probabilities should sum up to 1.
        """

        assert len(agent) == len(source) == self.Ndim

        # convert to numpy array for computation of d_over_dx and d_over_dx_downwind
        agent = np.array(agent)
        source = np.array(source)

        # dimensionless distance between source and agent, in number of gridcells.
        d_over_dx = np.linalg.norm(agent - source, ord=2)
        # print(f"d_over_dx = {d_over_dx}")

        # dimensionless downwind distance between source and agent used when V>0.
        # wind is taken to blow in positive 0-axis direction of the numpy array.
        # d_over_dx_downwind is positive when the agent is downwind from the source.
        d_over_dx_downwind = agent[0] - source[0]

        # get mean number of hits at d_over_dx
        mu = self._get_mu(flux, d_over_dx, d_over_dx_downwind)
        # print(f"mu at {d_over_dx} = {mu}")

        hits_max = self._get_hits_max(flux)

        hit_probabilities = np.zeros(self.Nhits)
        hit_probabilities[:hits_max] = self._Poisson(mu, range(hits_max))
        hit_probabilities[hits_max] = 1.0 - np.sum(
            hit_probabilities[: hits_max + 1]
        )  # correction such that sum of probabilities = 1.0
        
        # correction of negative hit probabilities.
        # this can happen for hit_max when some of the probabilities
        # that should be zero but are actually very small nubers.
        hit_probabilities[hit_probabilities < 0] = 0.0

        assert np.sum(hit_probabilities) == approx(1.0)
        assert np.any(hit_probabilities >= 0)
        
        return hit_probabilities

    def _get_hit_probabilities_from_meshgrid(self, agent, source, flux_index):
        """
        Look up the hit probabilities in hit_probabilities_meshgrid for a given
        agent location, source location and flux.

        Parameters 
        ----------
        agent: list, tuple or array of len Ndim
            agent location
        source: list, tuple or array of len Ndim
            true source location
        flux_index: int
            index of true source strength

        Returns
        -------
        hit_probabilities: array of size Nhits
            probability to observe number of hits based on the observation model.
            the probabilities should sum up to 1.
        """

        assert len(agent) == len(source) == self.Ndim

        loc_index = np.array([self.Ngrid] * self.Ndim) - np.array(agent) + np.array(source)
        assert len(loc_index) == self.Ndim
        # print(f"loc_index = {loc_index}")

        if self.Ndim == 1:
            hit_probabilities = self.hit_probabilities_meshgrid[
                ..., flux_index, loc_index[0]
            ]
        elif self.Ndim == 2:
            hit_probabilities = self.hit_probabilities_meshgrid[
                ..., flux_index, loc_index[0], loc_index[1]
            ]
        elif self.Ndim == 3:
            hit_probabilities = self.hit_probabilities_meshgrid[
                ...,
                flux_index,
                loc_index[0],
                loc_index[1],
                loc_index[2],
            ]

        assert len(hit_probabilities) == self.Nhits
        # print(f"hit_probabilities = {hit_probabilities}")

        return hit_probabilities


    def _get_observation(self, agent, source, flux, flux_index, seed):
        """
        Randomly draw an observation given the hit probabilities.
        """
        # get hit probabilities
        hit_probabilities = self._get_hit_probabilities_from_meshgrid(agent, source, flux_index)
        #print(f"hit_probabilities = {hit_probabilities}")

        # _get_hit_probabilities is more expensive than looking up value in meshgrid
        # but should give the same result
        # assert np.all(hit_probabilities == self._get_hit_probabilities(agent, source, flux))

        # get an observation based on the hit probabilities
        rng = np.random.default_rng(seed)
        observation = rng.choice(a=range(self.Nhits), p=hit_probabilities)

        return observation
