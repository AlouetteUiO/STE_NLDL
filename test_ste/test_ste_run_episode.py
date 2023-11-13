import numpy as np
import tensorflow as tf
import pytest
from pytest import approx

from ste_NLDL.classes.config import Config
from ste_NLDL.classes.training import TrainingEnv as Environment
from ste_NLDL.classes.observationmodel import VergassolaPlumeModel as ObservationModel
from ste_NLDL.classes.bayes import AnalyticalBayes as Bayes
from ste_NLDL.classes.neuralnetwork import ValueModel

from ste_NLDL.functions.run_episode import (
    run_one_episode,
    run_one_training_episode,
    run_batch_of_evaluation_episodes,
)

"""
Unit tests for run_episode
"""


def setup_1D_infotaxis():
    cfg = Config("test_ste/evaluate_infotaxis_1D_config.yml")
    from ste_NLDL.classes.training import POMDP as Environment
    from ste_NLDL.classes.policy import Infotaxis as Policy

    mod = ObservationModel(cfg)
    bay = Bayes(cfg, mod)
    env = Environment(cfg, mod, bay)
    pol = Policy(cfg, env, bay)
    return mod, bay, env, pol


def setup_1D_DRL():
    cfg = Config("test_ste/evaluate_DRL_1D_config.yml")
    from ste_NLDL.classes.training import TrainingEnv as Environment
    from ste_NLDL.classes.policy import DRLpolicy as Policy

    mod = ObservationModel(cfg)
    bay = Bayes(cfg, mod)
    env = Environment(cfg, mod, bay)
    NN = ValueModel(model_name='online_network', Ndim=1, FC_layers=8, FC_units=4, CNN_layers=0, CNN_filters=0, CNN_kernel_size=0, 
                    pooling=True, pooling_kernel_size = 2, pooling_strides = 2, regularization_factor = 0)
    pol = Policy(cfg, env)
    return mod, bay, env, pol, NN


def test_run_infotaxis():  # TODO This may give warning in inference_iteration
    """Checking if this works"""
    cfg = Config("test_ste/evaluate_infotaxis_1D_config.yml")
    from ste_NLDL.classes.training import POMDP as Environment
    from ste_NLDL.classes.policy import Infotaxis as Policy
    mod, bay, env, pol = setup_1D_infotaxis()
    # TODO improve this weird construction
    assert mod.Nhits == 5
    assert bay.Ndim == 1
    assert bay.Ngrid == 3
    assert env.mod.Nhits == 5
    assert env.Ndim == 1
    assert env.Ngrid == 3
    assert pol.bay.Ndim == 1
    env.max_t = 1
    memory = run_one_episode(env, pol, model=None, memorize=True)
    assert len(memory["states"]) == len(memory["observations"])
    # initial state observation is None
    assert memory["observations"][0] is None
    assert np.all(memory["states"][0].belief == 1 / env.Ngrid)
    


def test_run_DRL_train():
    """Checking if this works"""
    mod, bay, env, pol, model = setup_1D_DRL()
    assert mod.Nhits == 5
    assert env.mod.Nhits == 5
    env.max_t = 1
    # compile and build model (in learn.py)
    model.build_graph(belief_map_shape_nobatch=env.NN_input_shape)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    states_episode, next_states_episode, total_reward = run_one_training_episode(
        env, pol, model, eps=0.5
    )
    assert len(states_episode) == len(
        next_states_episode
    )  # the initial state and final state are not saved
