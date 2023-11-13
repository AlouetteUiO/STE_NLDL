import numpy as np
import pytest
from pytest import approx

from ste_NLDL.classes.config import Config
from ste_NLDL.classes.environment import POMDP as Environment
from ste_NLDL.classes.observationmodel import VergassolaPlumeModel as ObservationModel
from ste_NLDL.classes.bayes import AnalyticalBayes as Bayes

"""
Unit tests for class environment
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


def test_reset():
    mod, bay, env = setup_2D()
    assert env.Ngrid == 5
    assert env.Ndim == 2
    assert np.all(env.flux_range == np.array([1]))
    assert env.Nflux == 1
    init_belief, observation, info = env.reset()
    assert env.flux == 1
    assert env.belief.shape == (env.Nflux, env.Ngrid, env.Ngrid)
    assert np.sum(init_belief) == approx(1.0)
    assert np.all(init_belief == 1 / 25)
    assert np.sum(env.belief) == approx(1.0)
    env.set_agent = np.array([2, 2])
    env.set_source = np.array([2, 2])
    init_belief, observation, info = env.reset()
    assert np.all(env.agent == np.array([2, 2]))
    assert np.all(env.source == np.array([2, 2]))
    assert observation == env.mod.Nhits-1


def test_step():
    mod, bay, env = setup_2D()
    init_belief, observation, info = env.reset()
    assert env.t == 10
    step = 0
    terminated = False
    while terminated == False:
        (
            observation,
            terminated,
            reward,
            info
        ) = env.step(action=0)
        step += 1
        if step < 10:
            assert terminated == False
    assert env.t == 0
    assert step == 10


def test_check_converged_1D():
    mod, bay, env = setup_1D()
    # overwrite
    env.Ngrid = 5
    env.source = np.array([0])
    env.epsilon = 0.1
    env.belief = np.ones((1, 5)) / 5
    env.flux = 2
    assert np.all(env.belief == np.array([0.2, 0.2, 0.2, 0.2, 0.2]))
    converged, success_source, success_flux = env.check_converged()
    assert (
        converged == False
    )  # if not converged, then success_source and success_flux are False
    assert success_source == False
    assert success_flux == False
    env.belief = np.ones((1, 5)) * 0.01
    env.belief[0, 2] = 1 - 4 * 0.01
    assert np.all(env.belief == np.array([0.01, 0.01, 0.96, 0.01, 0.01]))
    converged, success_source, success_flux = env.check_converged()
    assert converged == True
    assert success_source == False
    assert success_flux == True
    env.belief = np.ones((1, 5)) * 0.01
    env.belief[0, 0] = 1 - 4 * 0.01
    assert np.all(env.belief == np.array([0.96, 0.01, 0.01, 0.01, 0.01]))
    converged, success_source, success_flux = env.check_converged()
    assert converged == True
    assert success_source == True
    assert success_flux == True


def test_move_1D():
    mod, bay, env = setup_1D()
    # overwrite
    env.Ngrid = 5
    env.Nactions = 3
    next_agent, move_possible = env.move(agent=np.array([4]), action=0)  # left
    assert next_agent == 3
    assert move_possible == True
    next_agent, move_possible = env.move(agent=np.array([0]), action=0)
    assert next_agent == 0
    assert move_possible == False
    next_agent, move_possible = env.move(agent=np.array([2]), action=1)  # right
    assert next_agent == 3
    assert move_possible == True
    next_agent, move_possible = env.move(agent=np.array([4]), action=1)
    next_agent = 4
    assert move_possible == False
    next_agent, move_possible = env.move(agent=np.array([4]), action=2)  # stay
    assert next_agent == 4
    assert move_possible == True


def test_move_2D():
    mod, bay, env = setup_2D()
    # overwrite
    env.Ngrid = 5
    env.Nactions = 5
    next_agent, move_possible = env.move(agent=np.array([4, 4]), action=0)  # up
    assert np.all(next_agent == np.array([3, 4]))
    assert move_possible == True
    next_agent, move_possible = env.move(agent=np.array([0, 0]), action=0)
    assert move_possible == False
    next_agent, move_possible = env.move(agent=np.array([2, 2]), action=1)  # down
    assert np.all(next_agent == np.array([3, 2]))
    assert move_possible == True
    next_agent, move_possible = env.move(agent=np.array([4, 2]), action=1)
    assert np.all(next_agent == np.array([4, 2]))
    assert move_possible == False
    next_agent, move_possible = env.move(agent=np.array([3, 3]), action=2)  # left
    assert np.all(next_agent == np.array([3, 2]))
    assert move_possible == True
    next_agent, move_possible = env.move(agent=np.array([1, 0]), action=2)
    assert move_possible == False
    next_agent, move_possible = env.move(agent=np.array([3, 3]), action=3)  # right
    assert np.all(next_agent == np.array([3, 4]))
    assert move_possible == True
    next_agent, move_possible = env.move(agent=np.array([1, 4]), action=3)
    assert move_possible == False
    next_agent, move_possible = env.move(agent=np.array([3, 3]), action=4)  # stay
    assert np.all(next_agent == np.array([3, 3]))
    assert move_possible == True


def test_transitions_1D():
    mod, bay, env = setup_1D()
    assert (
        env.Nactions == 2
    )  # stay = False, not possible for agent to stay in same grid cell
    current_belief = np.ones((env.Nflux, env.Ngrid), dtype=np.float32) / (
        env.Nflux * env.Ngrid
    )
    current_agent = np.array([env.Ngrid - 1])  # agent is at edge of domain
    env.belief = current_belief
    env.agent = current_agent
    env.t = 4
    state, next_states = env.transitions()
    ### check state
    assert state.belief.shape == (env.Nflux, env.Ngrid)
    assert np.all(state.belief == current_belief)
    assert np.all(state.agent == current_agent)
    assert state.prob == 1.0
    assert state.t == 4
    ### check next_states: array of State objects with size (Naction, Nhits)
    assert next_states.shape == (env.Nactions, env.mod.Nhits)
    assert len(next_states[0]) == len(next_states[1]) == env.mod.Nhits
    assert next_states[0][0].belief.shape == (env.Nflux, env.Ngrid)
    ### action 1 is to the right (outside the domain), the agent stays in the same grid cell
    assert np.all(  
        next_states[1][0].agent
        == next_states[1][1].agent
        == next_states[1][2].agent
        == next_states[1][3].agent
        == current_agent
    )
    ### action 0 is to the left
    assert np.all(  
        next_states[0][0].agent
        == next_states[0][1].agent
        == next_states[0][2].agent
        == next_states[0][3].agent
        == np.array([env.Ngrid - 2])
    )
    ### t of all next_states is 1 lower than state (count down to termination of episode)
    assert (
        next_states[0][0].t 
        == next_states[1][0].t 
        == next_states[0][1].t 
        == next_states[1][1].t 
        == next_states[0][2].t 
        == next_states[1][2].t 
        == next_states[0][3].t 
        == next_states[1][3].t 
        == 3
    ) 
    ### sum of belief is 1.0 in each State object
    assert (
        np.sum(next_states[0][0].belief)
        == np.sum(next_states[0][1].belief)
        == np.sum(next_states[0][2].belief)
        == np.sum(next_states[0][3].belief)
        == np.sum(next_states[0][4].belief)
        == 1.0
    )
    assert (
        np.sum(next_states[1][0].belief)
        == np.sum(next_states[1][1].belief)
        == np.sum(next_states[1][2].belief)
        == np.sum(next_states[1][3].belief)
        == np.sum(next_states[1][4].belief)
        == 1.0
    )
    ### sum of probabilities sums to 1.0 for each possible action
    assert (
        next_states[0][0].prob
        + next_states[0][1].prob
        + next_states[0][2].prob
        + next_states[0][3].prob
        + next_states[0][4].prob
    ) == approx(1.0)
    assert (
        next_states[1][0].prob
        + next_states[1][1].prob
        + next_states[1][2].prob
        + next_states[1][3].prob
        + next_states[1][4].prob
    ) == approx(1.0)
