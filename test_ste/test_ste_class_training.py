import numpy as np
import tensorflow as tf
import pytest
from pytest import approx

from ste_NLDL.classes.environment import State
from ste_NLDL.classes.config import Config
from ste_NLDL.classes.training import TrainingEnv as Environment
from ste_NLDL.classes.observationmodel import VergassolaPlumeModel as ObservationModel
from ste_NLDL.classes.bayes import AnalyticalBayes as Bayes
from ste_NLDL.classes.neuralnetwork import ValueModel
from ste_NLDL.functions.tools import get_entropy

"""
Unit tests for Training environment
"""


def setup_1D():
    cfg = Config("test_ste/evaluate_DRL_1D_config.yml")
    mod = ObservationModel(cfg)
    bay = Bayes(cfg, mod)
    env = Environment(cfg, mod, bay)
    NN = ValueModel(model_name='target_network', Ndim=1, FC_layers=8, FC_units=4, CNN_layers=0, CNN_filters=0, CNN_kernel_size=0, 
                    pooling=True, pooling_kernel_size = 2, pooling_strides = 2, regularization_factor = 0)
    return mod, bay, env, NN


def setup_2D():
    cfg = Config("test_ste/evaluate_DRL_2D_config.yml")
    mod = ObservationModel(cfg)
    bay = Bayes(cfg, mod)
    env = Environment(cfg, mod, bay)
    NN = ValueModel(model_name='target_network', Ndim=2, FC_layers=8, FC_units=4, CNN_layers=0, CNN_filters=0, CNN_kernel_size=0,
                    pooling=True, pooling_kernel_size = 2, pooling_strides = 2, regularization_factor = 0)
    return mod, bay, env, NN


def setup_3D():
    cfg = Config("test_ste/evaluate_DRL_3D_config.yml")
    mod = ObservationModel(cfg)
    bay = Bayes(cfg, mod)
    env = Environment(cfg, mod, bay)
    return mod, bay, env


def test_NN_input_shape():
    """
    NN_input_shape is defined in the init of TrainingEnv
    This is the shape of the input array for neural network models
    and the shape of the belief_map
    """
    mod, bay, env, NN = setup_1D()
    assert env.NN_input_shape == (env.Nflux, 5)
    mod, bay, env, NN = setup_2D()
    assert env.NN_input_shape == (env.Nflux, 5, 5)
    mod, bay, env = setup_3D()
    assert env.NN_input_shape == (env.Nflux, 5, 5, 5)


def test_states2inputs_1D_state():
    """
    Transforms the environments's state or possible next_states s' to input format
    the neural network model.
    Should be able to be used with:
    - different domain dimensions
    - states OR next_states
    - with or without batch_size
    The resulting input is the input for the neural network model
    and probs are the transition probabilities to different states
    """
    mod, bay, env, NN = setup_1D()
    # no batch
    env.belief = np.ones((env.Nflux, env.Ngrid), dtype=np.float32) / (env.Nflux * env.Ngrid)
    env.agent = np.array([1])
    env.t = 5
    state, next_states = env.transitions()
    input, probs, rewards = env.states2inputs(states=state)
    assert input.shape == (1, 1, 5) # batch_size, NN_input_shape
    assert probs.shape == rewards.shape == (1,) # batch_size -> no batch
    assert probs == 1.0
    assert rewards == 0.0
    # with batch
    batch_size = 3
    states = [0] * batch_size
    for i in range(batch_size):
        states[i] = state
    states = np.asarray(states)
    input, probs, rewards = env.states2inputs(states=states)
    assert input.shape == (3, 1, 5) # batch_size, NN_input_shape
    assert probs.shape == rewards.shape == (3,) # batch_size
    assert np.all(probs == 1)
    assert np.all(rewards == 0.0)

def test_states2inputs_2D_state():
    mod, bay, env, NN = setup_2D()
    # no batch
    env.belief = np.ones((env.Nflux, env.Ngrid, env.Ngrid), dtype=np.float32) / (
        env.Nflux * env.Ngrid * env.Ngrid
    )
    env.agent = np.array([1, 0])
    env.t = 5
    state, next_states = env.transitions()
    input, probs, rewards = env.states2inputs(states=state)
    assert input.shape == (1, 1, 5, 5) # batch_size, NN_input_shape
    assert probs.shape == rewards.shape == (1,) # batch_size -> no batch
    assert probs == 1.0
    assert rewards == 0.0
    # with batch
    batch_size = 3
    states = [0] * batch_size
    for i in range(batch_size):
        states[i] = state
    states = np.asarray(states)
    input, probs, rewards = env.states2inputs(states=states)
    assert input.shape == (3, 1, 5, 5) # batch_size, NN_input_shape
    assert probs.shape == rewards.shape == (3,) # batch_size
    assert np.all(probs == 1)
    assert np.all(rewards == 0.0)

def test_states2inputs_1D_next_states():
    mod, bay, env, NN = setup_1D()
    # no batch
    env.belief = np.ones((env.Nflux, env.Ngrid), dtype=np.float32) / (env.Nflux * env.Ngrid)
    env.agent = np.array([0])
    env.t = 5
    state, next_states = env.transitions()
    assert next_states.shape == (env.Nactions, env.mod.Nhits)
    assert len(next_states[0]) == len(next_states[1]) == env.mod.Nhits
    assert next_states[0][0].belief.shape == (env.Nflux, env.Ngrid)
    assert len(next_states[0][0].agent) == 1
    input, probs, rewards = env.states2inputs(states=next_states)
    assert input.shape == (1, 2, 5, 1, 5) # batch_size, Naction, Nhits, NN_input_shape
    assert probs.shape == rewards.shape == (1, env.Nactions, env.mod.Nhits)
    assert np.all(np.sum(probs, axis=(0,2)) == approx(1.0)) # sum along batch_size=1 and Nhits

def test_states2inputs_2D_next_states():
    mod, bay, env, NN = setup_2D()
    # no batch
    env.belief = np.ones((env.Nflux, env.Ngrid, env.Ngrid), dtype=np.float32) / (
        env.Nflux * env.Ngrid * env.Ngrid
    )
    env.agent = np.array([0, 1])
    env.t = 5
    state, next_states = env.transitions()
    input, probs, rewards = env.states2inputs(states=next_states)
    assert input.shape == (1, 4, 5, 1, 5, 5) # batch_size, Naction, Nhits, NN_input_shape
    assert probs.shape == (1, 4, 5)
    assert np.all(np.sum(probs, axis=(0,2)) == approx(1.0)) # sum along batch_size=1 and Nhits

def test_get_state_value():
    """ The model is build, but not trained """
    mod, bay, env, NN = setup_2D()
    NN.build_graph(belief_map_shape_nobatch=env.NN_input_shape)
    env.reset()
    state = State(env.belief, env.agent, env.t)
    value = env.get_state_value(NN, state)
    assert value.shape == (1,1)
    assert 0.0 <= value <= 0.1 # kernel is inialized between 0 and 0.1, see final_layer
    print(value)
    # with batch
    batch_size = 3
    states = [0] * batch_size
    for i in range(batch_size):
        states[i] = state
    values = env.get_state_value(NN, states)
    print(values)
    assert values.shape == (batch_size,1)
    assert np.all(value) == np.all(values)

def test_bellman():
    mod, bay, env, NN = setup_2D()
    # Assume batch_size = 1, Nactions = 1, Nhits = 2
    probs = tf.convert_to_tensor(np.array([[[0.5, 0.5]]], dtype=np.float64))
    rewards = tf.convert_to_tensor(np.array([[[2, 1]]], dtype=np.float64))
    next_values = tf.convert_to_tensor(np.array([[[10, 10]]], dtype=np.float64))
    assert next_values.shape == (1, 1, 2)
    expected_values = env.bellman_equation(probs, rewards, next_values)
    assert expected_values.shape == (1,1) 
    assert expected_values.numpy() == approx(11.5)
    # Assume batch_size = 1, Nactions = 3, Nhits = 2
    probs = tf.convert_to_tensor(np.array([[[0.5, 0.5], [0.9, 0.1], [0.7, 0.3]]], dtype=np.float64))
    rewards = tf.convert_to_tensor(np.array([[[2, 1], [2, 1], [2, 1]]], dtype=np.float64))
    next_values = tf.convert_to_tensor(np.array([[[10, 10], [10, 10], [10, 10]]], dtype=np.float64))
    assert next_values.shape == (1,3,2)
    expected_values = env.bellman_equation(probs, rewards, next_values)
    assert expected_values.shape == (1,3) 
    assert expected_values.numpy() == approx(np.array([[11.5, 11.9, 11.7]]))
    # Assume batch_size = 3, Nactions = 3, Nhits = 2
    probs = tf.convert_to_tensor(np.array([[[0.5, 0.5], [0.9, 0.1], [0.7, 0.3]], 
                                           [[0.5, 0.5], [0.9, 0.1], [0.7, 0.3]],
                                           [[0.5, 0.5], [0.9, 0.1], [0.7, 0.3]]], dtype=np.float64))
    rewards = tf.convert_to_tensor(np.array([[[2, 1], [2, 1], [2, 1]], 
                                             [[2, 1], [2, 1], [2, 1]],
                                             [[2, 1], [2, 1], [2, 1]]], dtype=np.float64))
    next_values = tf.convert_to_tensor(np.array([[[10, 10], [10, 10], [10, 10]], 
                                                 [[10, 10], [10, 10], [10, 10]],
                                                 [[10, 10], [10, 10], [10, 10]]], dtype=np.float64))
    assert next_values.shape == (3,3,2)
    expected_values = env.bellman_equation(probs, rewards, next_values)
    assert expected_values.shape == (3,3)
    assert expected_values.numpy() == approx(np.array([[11.5, 11.9, 11.7],
                                                       [11.5, 11.9, 11.7],
                                                       [11.5, 11.9, 11.7]]))


def test_get_target():
    mod, bay, env, NN = setup_2D()
    NN.build_graph(belief_map_shape_nobatch=env.NN_input_shape)
    env.reset()
    target_value = env.get_target(NN)
    assert target_value.shape == (1,1)
    

def test_centeragent_1D():
    mod, bay, env, NN = setup_1D()
    # overwrite
    agent = np.array([0])
    belief = np.arange(1, env.Ngrid + 1)
    belief_map = env._centeragent(belief, agent)
    print(f"belief_map = {belief_map}")
    # gives [[ 0.  0.  1.  2.  3. ]]
    assert np.all(belief_map[0, : env.Ngrid - 1] == 0)
    assert np.all(belief_map[0, env.Ngrid - 1 :] == belief)
    agent = np.array([env.Ngrid - 1])
    belief_map = env._centeragent(belief, agent)
    print(f"belief_map = {belief_map}")
    # gives [[ 1.  2.  3. 0. 0.]]
    assert np.all(belief_map[0, : env.Ngrid] == belief)
    assert np.all(belief_map[0, env.Ngrid :] == 0)


def test_centeragent_2D():
    mod, bay, env, NN = setup_2D()
    # overwrite
    agent = np.array([0, 0])
    belief = np.arange(1, env.Ngrid**2 + 1)
    belief = np.reshape(belief, newshape=(env.Ngrid, env.Ngrid))
    belief_map = env._centeragent(belief, agent)
    print(f"belief_map = {belief_map}")
    #[[[0. 0. 0. 0. 0.]
    #  [0. 0. 0. 0. 0.]
    #  [0. 0. 1. 2. 3.]
    #  [0. 0. 4. 5. 6.]
    #  [0. 0. 7. 8. 9.]]]
    assert np.all(belief_map[0, : env.Ngrid - 1, :] == 0)
    assert np.all(belief_map[0, : env.Ngrid, : env.Ngrid - 1] == 0)
    assert np.all(belief_map[0, env.Ngrid - 1 :, env.Ngrid - 1 :] == belief)
    agent = np.array([env.Ngrid - 1, env.Ngrid - 1])
    belief_map = env._centeragent(belief, agent)
    print(f"belief_map = {belief_map}")
    #[[[1. 2. 3. 0. 0.]
    #  [4. 5. 6. 0. 0.]
    #  [7. 8. 9. 0. 0.]
    #  [0. 0. 0. 0. 0.]
    #  [0. 0. 0. 0. 0.]]]
    assert np.all(belief_map[0, env.Ngrid :, :] == 0)
    assert np.all(belief_map[0, : env.Ngrid, env.Ngrid] == 0)
    assert np.all(belief_map[0, : env.Ngrid, : env.Ngrid] == belief)