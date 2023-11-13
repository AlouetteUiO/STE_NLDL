import numpy as np
from copy import deepcopy
from ste_NLDL.functions.tools import get_DRPS


class State:
    """
    Defines a belief state of the environment.
    Used to store transitions (s, s')

    Attributes
    ----------
    belief: array of shape tuple([Nflux] + [Ngrid] * Ndim)
        probability distribution of the source
    agent: array of shape tuple([Ndim])
        location of the agent
    t: int
        number of timesteps left until end of episode  # NOTE legacy code
    prob: scalar (float)
        - if current state s, then prob = 1.0
        - if next state s', then prob = probability to transit from s to this state
    reward: scalar (float)
        - if current state s, then reward = 0.0
        - if next state s', then reward for transit from s to this state
    """

    def __init__(self, belief, agent, t, prob=1.0, reward=0.0):
        self.belief = np.asarray(belief, dtype=np.float32)
        self.agent = agent
        self.t = t
        self.prob = prob
        self.reward = reward


class POMDP:
    """
    Environment to simulate the estimation of the source location and strength of a hotspot.
    This is a POMDP (Partially Observable Markov Decision Process) environment.
    """

    def __init__(self, cfg, mod, bay):
        """
        Initialize the environment
        """
        self.mod = mod
        self.bay = bay

        self.index = cfg.get("methods/environment")

        if self.index == 0:
            self.name = "POMDP"
        elif self.index == -1:
            self.name = "RL Training POMDP"

        # Set mandatory domain variables
        self.Ndim = int(cfg.get("domain/Ndim"))
        self.Ngrid = int(cfg.get("domain/Ngrid"))
        assert self.Ngrid % 2 == 1  # Ngrid should be uneven
        self.V = float(cfg.get("domain/V"))  # non-dimensional wind speed
        self.D = float(cfg.get("domain/D"))  # non-dimensional diffusivity
        self.belief_criteria = float(cfg.get("domain/belief_criteria"))
        self.max_t = int(cfg.get("nn/stop_t"))

        # Set optional domain variables
        self.set_agent = cfg.get(
            "domain/agent"
        )  # if None, the agent starts at center of domain (function reset)
        self.set_source = cfg.get(
            "domain/source"
        )  # if None, the source is drawn randomly (function reset)
        self.set_flux = cfg.get(
            "domain/flux"
        )  # if None, the flux is drawn randomly from flux_range (function reset)
        self.flux_range = np.array(cfg.get("domain/flux_range"))
        self.Nflux = len(self.flux_range)

        # Set other variables
        stay = cfg.get("domain/stay")
        if stay:
            self.Nactions = 2 * self.Ndim + 1  # agent can choose to stay in node
        else:
            self.Nactions = 2 * self.Ndim  # agent has to move to another node

    def reset(self, seed=None):
        """
        Reset the environment
        1. Initialize the agent.
        2. Initialize the source location.
        3. Initialize the flux.
        4. Initialize the belief = distribution over the lattice. Start with a uniform distribution.
        5. Get an observation at the agent location.
        6. Update the belief using Bayesian inference.
        7. Get info.
        """

        # Initialize agent. If none is given, start the agent at the center of the domain
        if self.set_agent is None:
            self.agent = np.array([self.Ngrid // 2] * self.Ndim)
        else:
            self.agent = np.array(self.set_agent)
        # print(f"agent = {self.agent}")

        # Initialize source location
        if self.set_source is None:
            rng = np.random.default_rng()
            self.source = rng.integers(low=0, high=self.Ngrid, size=self.Ndim)
        else:
            self.source = np.array(self.set_source)
        # print(f"source = {self.source}")

        # Initialize the flux
        if self.set_flux is None:
            rng = np.random.default_rng()
            self.flux_index = rng.integers(self.Nflux)
            self.flux = self.flux_range[self.flux_index]
        else:
            self.flux = np.array(self.set_flux)
            self.flux_index = int(np.argwhere(self.flux_range == self.flux))
        # print(f"self.flux = {self.flux}")
        # print(f"self.flux_index = {self.flux_index}")

        self.t = self.max_t

        # Initialize the belief of the agent: a uniform distribution over the lattice
        init_belief = np.ones([self.Nflux] + [self.Ngrid] * self.Ndim) / (
            self.Nflux * self.Ngrid**self.Ndim
        )
        # print(f"belief = \n{init_belief}")

        # Get true probability density function
        self.pdf_true = np.zeros_like(init_belief)
        self.pdf_true[(self.flux_index,) + tuple(self.source)] = 1.0

        # Get observation at starting position of the agent
        observation = self.mod.get_observation(self.agent, self.source, self.flux, self.flux_index, seed)
        # print(f"observation = {observation}")

        # Update belief of the agent
        self.belief, _, _ = self.bay.inference_iteration(
            init_belief, self.agent, observation
        )
        # print(f"belief = \n{self.belief}")

        info = self.get_info()

        return init_belief, observation, info

    def step(self, action, seed=None):
        """
        0. Update t
        1. The agent moves to a new position following 'action'.
        2. Get an observation at the agent location.
        3. Update the belief using Bayesian inference.
        4. Get a reward.
        5. Get info.
        """

        self.t -= 1
        # print(f"t = {self.t}")

        # execute_action
        self.agent, _ = self.move(self.agent, action)
        # print(f"new location of agent = {self.agent}")

        # get observation at new position
        observation = self.mod.get_observation(self.agent, self.source, self.flux, self.flux_index, seed)
        # print(f"observation = {observation}")

        # Update belief of the agent
        self.belief, entropy, _ = self.bay.inference_iteration(
            self.belief, self.agent, observation
        )
        # print(f"belief = \n{self.belief}")
        # print(f"belief at agent location = {self.belief[tuple(self.agent)]}")

        terminated = True if self.t == 0 else False

        reward = entropy

        info = self.get_info()

        return observation, terminated, reward, info
    
    def get_info(self):
        """
        Get info on:
        1. if source has been found, and if the source has been identified correctly. 
        2. DRPS and relative DRPS of the current belief

        Returns
        -------
        info: dict
            contains info on convergence and statistics of the belief state.
        """

        # Check if a source has been identified (successfully)
        converged, success_source, success_flux = self.check_converged()

        # Get DRPS and relative DRPS of the distribution
        DRPS, relDRPS = get_DRPS(self.belief, self.pdf_true)

        info = {"converged": converged,
                "success_source": success_source,
                "success_flux": success_flux,
                "DRPS_q": DRPS[0],
                "DRPS_x": DRPS[1],
                "DRPS_qrel": relDRPS[0],
                "DRPS_xrel": relDRPS[1],
                }
        if self.Ndim > 1:
            info["DRPS_y"] = DRPS[2]
            info["DRPS_yrel"] = relDRPS[2]

        return info


    def check_converged(self):
        """
        Check if source has been found, and if the source has been identified correctly.

        Returns
        -------
        converged: boolean
            True if the belief is >= belief_criteria on one of the nodes of the lattice, else False.
        success_source: boolean
            True if converged and the highest belief is on the node of the true source, else False.
        success_flux: boolean
            True if converged and the highest belief is on the node of the true flux, else False
        """
        converged = False
        success_source = False
        success_flux = False

        if np.any(self.belief >= self.belief_criteria):
            converged = True
            index_belief = np.unravel_index(
                np.argmax(self.belief, axis=None), self.belief.shape
            )

            index_source = index_belief[1:]
            if np.all(self.source == index_source):
                success_source = True

            index_flux = index_belief[0]
            if self.flux == self.flux_range[index_flux]:
                success_flux = True

        return converged, success_source, success_flux

    def move(self, agent, action):
        """
        The agent executes an action.

        Parameters
        ----------
        agent: array
            location of the agent
        action: int
            action corresponding to direction of movement or staying in grid cell

        Returns
        -------
        next_agent: array
            next location of the agent
        move_possible: bool
            move_possible is False if the agent would step outside the domain,
            otherwise True.
        """
        next_agent = deepcopy(agent)
        move_possible = True
        if action == self.Ndim * 2:  # stay
            pass
        elif action < self.Ndim * 2:  # move
            axis = action // 2
            direction = 2 * (action % 2) - 1  # +1 or -1
            if direction == -1:
                if agent[axis] > 0:
                    next_agent[axis] -= 1
                else:
                    move_possible = False
            elif direction == 1:
                if agent[axis] < self.Ngrid - 1:
                    next_agent[axis] += 1
                else:
                    move_possible = False
        else:
            raise Exception("This action is outside range")
        return next_agent, move_possible

    def transitions(self):
        """
        Returns all possible next states s' that can be reached from state s,
        given all possible actions and hits (observations).

        NOTE: If agent is at the edge of the domain and the action
        would take the agent outside of the domain,
        the agent stays in the same location in next_states for this action.

        Returns
        -------
        state: State object
            current state of the agent
        next_states: array of State objects with size (Naction, Nhits)
            array of all possible next states of the agent for all possible actions and hit values
        """

        state = State(self.belief, self.agent, self.t, prob=1.0)

        next_states = np.empty(shape=(self.Nactions, self.mod.Nhits), dtype=State)
        for action in range(self.Nactions):

            next_agent, _ = self.move(self.agent, action)

            prior = deepcopy(self.belief)
            # NOTE: inference iteration without observation
            posterior, entropy, prob = self.bay.inference_iteration(prior, next_agent)

            for h in range(self.mod.Nhits):
                # print(f"action = {action}, h = {h}, prob = {prob[h]}, reward = {entropy[h]}")
                next_state = State(
                    posterior[h], next_agent, self.t - 1, prob[h], reward=entropy[h]
                )
                next_states[action, h] = next_state

        return state, next_states
