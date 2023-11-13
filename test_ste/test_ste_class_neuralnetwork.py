"""
Test the ValueModel class
"""

import tensorflow as tf
import numpy as np
import pytest
from pytest import approx

from ste_NLDL.classes.config import Config
from ste_NLDL.classes.training import TrainingEnv as Environment
from ste_NLDL.classes.observationmodel import VergassolaPlumeModel as ObservationModel
from ste_NLDL.classes.bayes import AnalyticalBayes as Bayes
from ste_NLDL.classes.neuralnetwork import ValueModel


def setup_1D():
    cfg = Config("test_ste/evaluate_DRL_1D_config.yml")
    mod = ObservationModel(cfg)
    bay = Bayes(cfg, mod)
    env = Environment(cfg, mod, bay)
    model = ValueModel(model_name='test', Ndim=1, FC_layers=8, FC_units=4, CNN_layers=0, CNN_filters=1, CNN_kernel_size=1, pooling=True,
    pooling_kernel_size = 2, pooling_strides = 2, regularization_factor = 0)
    # TODO if we use env as input to NNModel, we can access env.NN_input_shape

    # compile and build model (in learn.py)
    model.build_graph(belief_map_shape_nobatch=env.NN_input_shape)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

    return mod, bay, env, model


def setup_2D():
    cfg = Config("test_ste/evaluate_DRL_2D_config.yml")
    mod = ObservationModel(cfg)
    bay = Bayes(cfg, mod)
    env = Environment(cfg, mod, bay)
    model = ValueModel(model_name='test', Ndim=2, FC_layers=8, FC_units=4, CNN_layers=0, CNN_filters=1, CNN_kernel_size=1, pooling=True,
    pooling_kernel_size = 2, pooling_strides = 2, regularization_factor = 0)

    # compile and build model (in learn.py)
    model.build_graph(belief_map_shape_nobatch=env.NN_input_shape)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

    return mod, bay, env, model


def test_call_1D():
    mod, bay, env, model = setup_1D()
    env.belief = np.ones((env.Nflux, env.Ngrid), dtype=np.float32) / (env.Nflux * env.Ngrid)
    env.agent = np.array([1])
    env.t = 5
    state, next_states = env.transitions()
    input, probs, _ = env.states2inputs(states=state)
    assert input.shape == (1, 1, 5) # batch_size = 1, Nflux = 1, and 1D grid of 5 grid cells
    assert model.call(x=input, training=False) == model.call(
        x=tf.convert_to_tensor(input), training=False
    )
    x = model.call(x=input, training=False)
    assert x.shape == (1, 1)
    x = model.call(x=input, training=True)
    assert x.shape == (1, 1)


def test_call_2D():
    mod, bay, env, model = setup_2D()
    env.belief = np.ones((env.Nflux, env.Ngrid, env.Ngrid), dtype=np.float32) / (
        env.Nflux * env.Ngrid * env.Ngrid
    )
    env.agent = np.array([1, 1])
    env.t = 5
    state, next_states = env.transitions()
    input, probs, _ = env.states2inputs(states=state)
    assert input.shape == (1, 1, 5, 5) # batch_size, Nflux, grid
    assert model.call(x=input, training=False) == model.call(
        x=tf.convert_to_tensor(input), training=False
    )
    x = model.call(x=input, training=False)
    assert x.shape == (1, 1)
    x = model.call(x=input, training=True)
    assert x.shape == (1, 1)


def test_train_step():
    mod, bay, env, model = setup_1D()
    env.belief = np.ones((env.Nflux, env.Ngrid), dtype=np.float32) / (env.Nflux * env.Ngrid)
    env.agent = np.array([1])
    env.t = 5
    state, next_states = env.transitions()
    input, probs, _ = env.states2inputs(states=state)
    loss = model.train_step(x=input, y=tf.convert_to_tensor(10))
    assert loss.shape == ()  # one value


def test_test_step():
    mod, bay, env, model = setup_1D()
    env.belief = np.ones((env.Nflux, env.Ngrid), dtype=np.float32) / (env.Nflux * env.Ngrid)
    env.agent = np.array([1])
    env.t = 5
    state, next_states = env.transitions()
    input, probs, _ = env.states2inputs(states=state)
    loss = model.test_step(x=input, y=tf.convert_to_tensor(10))
    assert loss.shape == ()  # one value
