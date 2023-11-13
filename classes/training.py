import tensorflow as tf
import numpy as np
from copy import deepcopy

from ste_NLDL.classes.environment import POMDP


class TrainingEnv(POMDP):
    """
    TrainingEnv is a subclass of the POMDP superclass
    This class is used to train the agent in deep reinforcement learning
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, **kwargs
        )

        self.NN_input_shape = tuple([self.Nflux] + [2 * self.Ngrid - 1] * self.Ndim)
        # (1, 9, 9) for Nflux = 1, Ngrid = 5, Ndim = 2

    def states2inputs(self, states):
        """
        Converts states into the input format used by the neural network model.

        Parameters
        ----------
        states: State object or NumPy array of State objects
            if current state s: single State or -> states.ndim == 0
                                NumPy array of State objects with shape batch_size -> states.ndim == 1
            if next states s':  NumPy array of State objects with shape (Nactions, Nhits) or -> states.ndim == 2
                                NumPy array of State objects with shape (batch_size, Nactions, Nhits) -> states.ndim == 3

        Returns
        -------
        inputs: TensorFlow tensor
            tensor of inputs for the neural network model with:
            if current state s: shape (batch_size, input_shape)
            if next states s':  shape (batch_size, Nactions, Nhits, input_shape)
            with input_shape: tuple([Nflux] + [2 * Ngrid - 1] * Ndim)
        probs: TensorFlow tensor
            if current state s: tensor of 1.0 with shape (batch_size)
            if next states s':  tensor of transition probabilities with shape (batch_size, Nactions, Nhits)
        rewards: TensorFlow tensor
            if current state s: tensor of None with shape (batch_size)
            if next states s':  tensor of rewards with shape (batch_size, Nactions, Nhits)
        """

        states = np.array(states)
        if states.ndim == 0:  # state without batch_size, set batch_size = 1
            states = states[np.newaxis]
        if states.ndim == 2:  # next states without batch_size, set batch_size = 1
            states = states[np.newaxis, :]

        assert (
            states.ndim == 1 or states.ndim == 3
        )  # ndim == 1 for state and ndim == 3 for next_states
        if states.ndim == 3:
            assert states.shape[1:] == (self.Nactions, self.mod.Nhits)

        inputs = np.asarray([self._state2input(state) for state in states.ravel()])
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        inputs = tf.reshape(inputs, shape=states.shape + self.NN_input_shape)

        probs = np.asarray([state.prob for state in states.ravel()])
        probs = tf.convert_to_tensor(probs, dtype=tf.float32)
        probs = tf.reshape(probs, shape=states.shape)

        rewards = np.asarray([state.reward for state in states.ravel()])
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        rewards = tf.reshape(rewards, shape=states.shape)

        return inputs, probs, rewards

    def _state2input(self, state):
        """
        Transform the belief and location of the agent to a probability map centered
        around the location of the agent. This is the input format for the neural
        network model.

        Parameters
        ----------
        state: State object
            current state or possible next state of the environment

        Returns
        -------
        input: NumPy array of length NN_input_shape
            input for neural network model
        """
        belief_map = self._centeragent(state.belief, state.agent)
        input = np.array(belief_map, dtype=np.float32) 
        assert input.shape == self.NN_input_shape

        return input

    def _centeragent(self, belief, agent):
        """
        Returns a 'belief_map' of the location of the source centered on the agent.

        Parameters
        ----------
        belief: NumPy array
            belief of the location of the source (and strength) in a non-centered environment
        agent: NumPy array
            location of the agent

        Returns
        -------
        belief_map: NumPy array (shape is NN_input_shape)
            belief of the location of the source centered around the location of the agent
            Example: with Nflux = 1, Ngrid = 3 and Ndim = 2, belief_map has shape (1,5,5)
        """

        belief_map = np.zeros(self.NN_input_shape)
        if self.Ndim == 1:
            belief_map[
                :, self.Ngrid - 1 - agent[0] : 2 * self.Ngrid - 1 - agent[0]
            ] = belief
        elif self.Ndim == 2:
            belief_map[
                :,
                self.Ngrid - 1 - agent[0] : 2 * self.Ngrid - 1 - agent[0],
                self.Ngrid - 1 - agent[1] : 2 * self.Ngrid - 1 - agent[1],
            ] = belief
        elif self.Ndim == 3:
            belief_map[
                :,
                self.Ngrid - 1 - agent[0] : 2 * self.Ngrid - 1 - agent[0],
                self.Ngrid - 1 - agent[1] : 2 * self.Ngrid - 1 - agent[1],
                self.Ngrid - 1 - agent[2] : 2 * self.Ngrid - 1 - agent[2],
            ] = belief
        else:
            raise Exception("_centeragent is not implemented for Ndim > 3")

        return belief_map

    def bellman_equation(self, probs, rewards, next_values):
        """
        Bellman equation with gamma = 1 for the approximate value function
        following policy pi (by neural network model).

        q_pi(s,a) = SUM_{s',r} p(s',r|s,a) [reward + gamma * v_pi(s')]

        Parameters
        ----------
        probs: TensorFlow tensor with shape (batch_size, Nactions, Nhits)
            state transition probabilities p(s',r|s,a)
        rewards: TensorFlow tensor with shape (batch_size, Nactions, Nhits)
            rewards r
        next_values: TensorFlow tensor with shape (batch_size, Nactions, Nhits)
            next state values v_pi(s')

        Returns
        --------
        action_values: TensorFlow tensor with shape (batch_size, Nactions)
            state action values q_pi(s,a)
        """
        assert next_values.ndim == 3  # batch_size, Nactions, Nhits
        assert next_values.shape == probs.shape == rewards.shape
        action_values = tf.math.reduce_sum(probs * (rewards + next_values), axis=-1)

        return action_values

    def get_state_value(self, model, state):
        """
        Returns state value v(s) given by current policy pi by the (neural network) model.

        Parameters
        ----------
        model: ValueModel object
            neural network model
        state: single State object or NumPy array of State objects with shape (batch_size, )
            current state(s)

        Returns
        -------
        value: NumPy array with shape (batch_size, 1)
            value of state(s) according to model
        """

        input, _, _ = self.states2inputs(state)
        value = model(input, training=False)
        return value.numpy()

    def get_action_values(self, model, next_states):
        """
        Returns action values q_pi(s,a) given current policy pi by the (neural network) model
        and the values of the next states v_pi(s') that can be reached from the current state s.

        Parameters
        ----------
        model: ValueModel object
            neural network model
        next_states: NumPy array of State objects with shape (Nactions, Nhits) or (batch_size, Nactions, Nhits)
            all next states s' that can be reached from state s for all possible actions and hit values.

        Returns
        -------
        action_values: TensorFlow tensor with shape (batch_size, Nactions)
            action values q_pi(s,a) for state s and all possible actions a.
        """

        if next_states.ndim == 2:  # no batch_size given
            batch_size = 1
        else:
            batch_size = next_states.shape[0]
        Nactions = self.Nactions
        Nhits = self.mod.Nhits

        inputs, probs, rewards = self.states2inputs(next_states)
        assert inputs.shape == tuple(
            [batch_size] + [Nactions] + [Nhits] + list(self.NN_input_shape) 
        )
        assert probs.shape == tuple([batch_size] + [Nactions] + [Nhits])
        assert rewards.shape == tuple([batch_size] + [Nactions] + [Nhits])

        # reshape inputs
        inputs = tf.reshape(
            inputs,
            shape=tuple([batch_size * Nactions * Nhits] + list(self.NN_input_shape)), 
        )  # (batch_size * Nactions * Nhits, NN_input_shape)

        next_values = model(inputs, training=False)
        assert next_values.shape == tuple([batch_size * Nactions * Nhits] + [1])
        # print(f"next_values = \n{next_values}")

        next_values = tf.reshape(
            next_values,
            shape=(
                batch_size,
                Nactions,
                Nhits,
            ),
        )
        assert next_values.shape == tuple([batch_size] + [Nactions] + [Nhits])
        # print(f"next_values reshaped = \n{next_values}")

        action_values = self.bellman_equation(probs, rewards, next_values)
        assert action_values.shape == tuple(
            [batch_size] + [Nactions]
        )  # (batch_size, Nactions)
        # print(f"action_values = \n{action_values}")

        return action_values

    def get_target(self, model, next_states=None):
        """
        Compute the target value (for training) of a state s.
        This is the estimated optimal value for state s.

        v_*(s) = MIN_a q_{pi_*}(s,a)

        NOTE that we use the minimum as we aim to mimimize
        the expected return (based on sum of entropy) from the state.

        Parameters
        ----------
        model: ValueModel object
            neural network model
        next_states: array of State objects with shape (Nactions, Nhits) or (batch_size, Naction, Nhits) (optional)
            all values of next states s' that can be reached from state s by all possible actions and hit values.

        Returns
        -------
        target: array with shape (batch_size, 1)
            target value(s) of state s according to neural network model
        """

        assert model.model_name == 'target_network'

        if next_states is None:
            _, next_states = self.transitions()

        if next_states.ndim == 2:
            next_states = next_states[np.newaxis, :]

        batch_size = next_states.shape[0]
        # print(f"batch_size = {batch_size}")

        # Get the value of being in state s and performing action 0, 1, 2, etc.
        action_values = self.get_action_values(model, next_states)
        assert action_values.shape == tuple([batch_size] + [self.Nactions])
        # print(f"action_values in get_target = \n{action_values}")

        # Get target
        target = tf.math.reduce_min(action_values, axis=1, keepdims=True)
        assert target.shape == tuple([batch_size] + [1])
        # print(f"target = {target}")

        return target
