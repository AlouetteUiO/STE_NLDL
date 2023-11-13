import numpy as np
import tensorflow as tf
from copy import deepcopy

from ste_NLDL.functions.tools import get_entropy


class Policy:
    """
    Policy superclass
    """

    def __init__(self, cfg, env):
        self.index = cfg.get("methods/policy")

        if self.index == -1:
            self.name = "neural network"
        elif self.index == 0:
            self.name = "infotaxis"

        self.env = env

    def select_epsilon_greedy_action(self, model=None, epsilon=0.0):
        """
        Select next action for agent following an epsilon-greedy policy.
        A random action is selected with probability epsilon.
        The best action is chosen with probability 1 - epsilon.

        Arguments
        ---------
        epsilon: float (between 0.0 and 1.0)
            exploration rate of the agent, ranging from
            0.0 = greedy action selection (exploitation)
            1.0 = random action selection (exploration)
        Returns
        -------
        action: int
            selected action for a move that is possible
        """

        rng = np.random.default_rng()
        move_possible = False
        while not move_possible:
            if rng.random() < epsilon:  # exploration
                action = rng.choice(self.env.Nactions)
            else:  # exploitation
                action = self._select_best_action(model)
            _, move_possible = self.env.move(self.env.agent, action)
        return action

    def select_random_action(self):
        """Similar to select_epsilon_greedy_action with epsilon = 1.0.
        Full exploration."""

        move_possible = False
        while not move_possible:
            rng = np.random.default_rng()
            action = rng.choice(self.env.Nactions)
            _, move_possible = self.env.move(self.env.agent, action)
        return action

    def select_best_action(self, model=None):
        """Similar to select_epsilon_greedy_action with epsilon = 0.0.
        Full exploitation."""

        action = self._select_best_action(model)
        _, move_possible = self.env.move(self.env.agent, action)

        if move_possible == False:
            while not move_possible:
                rng = np.random.default_rng()
                action = rng.choice(self.env.Nactions)
                _, move_possible = self.env.move(self.env.agent, action)

        return action


class Infotaxis(Policy):
    """
    Infotaxis subclass
    This is a child class from Policy

    Original infotaxis from Vergassola, adapted to source
    term estimation instead of source seeking only.

    For each candidate action:
    1. Move agent following the candidate action
    2. Get observation following current belief
    3. Compute candidate information gain

    Select action with highest information gain
    """

    def __init__(self, cfg, env, bay):
        self.bay = bay

        super().__init__(cfg, env)

    def _select_best_action(self, model):
        assert model == None

        delta_entropy = np.zeros(self.env.Nactions)

        for action in range(self.env.Nactions):
            # print(f"\naction = {action}\n")

            prior = deepcopy(
                self.env.belief
            )  # NOTE: This has to be done for each action, otherwise prior is used from previous iteration
            prior_entropy = get_entropy(probability_distributon=prior, axis=None)
            # print(f"--- prior = \n{prior}")
            # print(f"--- prior entropy = {prior_entropy}")

            # move agent
            next_agent, move_possible = self.env.move(self.env.agent, action)

            if move_possible:
                _, entropy, probability = self.bay.inference_iteration(
                    prior, next_agent, observation=None
                )
                # entropy.shape == probability.shape == (Nhits,)
                entropy = np.sum(probability * entropy)
                delta_entropy[action] = prior_entropy - entropy
            else:
                delta_entropy[action] = np.NaN

        # print(f"delta entropy = {delta_entropy}")
        action = np.nanargmax(delta_entropy)
        # print(f"--- selected action following infotaxis = {action}")

        return action


class DRLpolicy(Policy):
    """
    Deep reinforcement learning policy subclass
    This is a child class from Policy

    This policy is based on a value model trained with a neural network.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)

    def _select_best_action(self, model):
        """
        Select the action that minimizes the value function.

        v_*(s) = MIN_a q_{pi_*}(s,a)

        NOTE that we use the minimum as we aim to mimimize
        the expected return (based on sum of entropy) from the state.
        """

        assert model.model_name == "online_network"
        _, next_states = self.env.transitions()
        assert next_states.ndim == 2  # no batch_size
        action_values = (
            self.env.get_action_values(  # TODO here is goes wrong in parallel
                model, next_states
            )
        )

        action_values = tf.reshape(action_values, shape=(self.env.Nactions,))
        action = tf.math.argmin(action_values)

        return action
