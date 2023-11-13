import numpy as np
import multiprocessing as mp
from itertools import repeat
from ste.classes.environment import State
from ste.functions.tools import get_entropy


def run_one_episode(env, pol, model, source=None, flux=None, memorize=False, seed=None):
    """
    Execute one episode: one trajectory of the agent.
    The agent chooses the best actions.

    Arguments
    ---------
    env: POMDP or TrainingEnv object
    pol: Policy object
    memorize: boolean 
        whether to store the trajectory in a dict

    Returns
    -------
    memory: dict
        contains states and observations of the episode
    """

    memory = {"states": [], "ref_next_agent": [], "observations": [], "entropy": []}
    
    env.set_source = source
    env.set_flux = flux
    init_belief, observation, info = env.reset(seed)
    print(f"observation = {observation}")
    if memorize:
        memory["observations"].append(None)
        memory["observations"].append(observation)
        memory["states"].append(State(belief = init_belief, agent = env.agent, t = env.t))
        memory["states"].append(State(belief = env.belief, agent = env.agent, t = env.t))
        memory["source"] = env.source
        memory["flux"] = env.flux
        memory["entropy"].append(get_entropy(init_belief))
        memory["entropy"].append(get_entropy(env.belief))
        memory["ref_next_agent"].append(None)
        memory["ref_next_agent"].append(None)

    terminated = False
    while terminated == False:

        action = pol.select_best_action(model)
        #print(f"action = {action}")
        
        observation, terminated, reward, info = env.step(action, seed)
        # print(f"entropy = {reward}")

        if memorize:
            memory["observations"].append(observation)
            memory["states"].append(State(env.belief, env.agent, env.t))
            memory["entropy"].append(get_entropy(env.belief))

    env.set_source = None    
    env.set_flux = None
    return memory


def run_one_random_episode(env, pol, memorize=False):
    """
    Execute one episode: one trajectory of the agent.
    The agent chooses only random actions.

    Arguments
    ---------
    env: POMDP or TrainingEnv object
    pol: Policy object
    memorize: boolean 
        whether to store the trajectory in a dict

    Returns
    -------
    memory: dict
        contains states and observations of the episode
    """

    memory = {"states": [], "observations": []}
    
    init_belief, observation, _, _, _, _ = env.reset()
    if memorize:
        memory["observations"].append(None)
        memory["observations"].append(observation)
        memory["states"].append(State(belief = init_belief, agent = env.agent))
        memory["states"].append(State(belief = env.belief, agent = env.agent))
        memory["source"] = env.source

    terminated = False
    while terminated == False:

        action = pol.select_random_action()
        observation, terminated, _, _, _, _, _ = env.step(action)

        if memorize:
            memory["observations"].append(observation)
            memory["states"].append(State(env.belief, env.agent))

    return memory

def run_batch_of_evaluation_episodes(filename, batch_size, env, pol, model=None):
    """
    Execute batch of episodes for policy evaluation and saves the results to .npy files.

    Arguments
    ---------
    filename: str
    batch_size: int
        number of episodes
    env: POMDP or TrainingEnv object
    pol: Policy object
    """

    print(f"---- Run_batch_of_evaluation_episodes for policy {pol} ----")

    reward_batch = np.zeros(shape=(batch_size, env.max_t))
    converged_batch = np.zeros(shape=(batch_size, env.max_t))
    success_source_batch = np.zeros(shape=(batch_size, env.max_t))
    success_flux_batch = np.zeros(shape=(batch_size, env.max_t))
    DRPS_q_batch = np.zeros(shape=(batch_size, env.max_t))
    DRPS_x_batch = np.zeros(shape=(batch_size, env.max_t))
    DRPS_y_batch = np.zeros(shape=(batch_size, env.max_t))
    DRPS_qrel_batch = np.zeros(shape=(batch_size, env.max_t))
    DRPS_xrel_batch = np.zeros(shape=(batch_size, env.max_t))
    DRPS_yrel_batch = np.zeros(shape=(batch_size, env.max_t))

    for i_episode in range(batch_size):
    
        init_belief, observation, info = env.reset()

        terminated = False
        step = 0
        while terminated == False:

            state, next_states = env.transitions()

            action = pol.select_best_action(model) 
            observation, terminated, reward, info = env.step(action)

            reward_batch[i_episode, step] = reward
            converged_batch[i_episode, step] = info["converged"]
            success_source_batch[i_episode, step] = info["success_source"]
            success_flux_batch[i_episode, step] = info["success_flux"]
            DRPS_q_batch[i_episode, step] = info["DRPS_q"] 
            DRPS_x_batch[i_episode, step] = info["DRPS_x"] 
            DRPS_y_batch[i_episode, step] = info["DRPS_y"]  
            DRPS_qrel_batch[i_episode, step] = info["DRPS_q"] 
            DRPS_xrel_batch[i_episode, step] = info["DRPS_xrel"] 
            DRPS_yrel_batch[i_episode, step] = info["DRPS_yrel"] 
            step += 1

    np.save(filename + "_reward.npy", reward_batch)
    np.save(filename + "_converged.npy", converged_batch)
    np.save(filename + "_success_source.npy", success_source_batch)
    np.save(filename + "_success_flux.npy", success_flux_batch)
    np.save(filename + "_DRPS_q.npy", DRPS_q_batch)
    np.save(filename + "_DRPS_x.npy", DRPS_x_batch)
    np.save(filename + "_DRPS_y.npy", DRPS_y_batch)
    np.save(filename + "_DRPS_qrel.npy", DRPS_qrel_batch)
    np.save(filename + "_DRPS_xrel.npy", DRPS_xrel_batch)
    np.save(filename + "_DRPS_yrel.npy", DRPS_yrel_batch)
    
    return None


def run_one_training_episode(env, pol, model, eps):
    """
    Execute one episode to train neural network model.

    Arguments
    ---------
    env: TrainingEnv object
    pol: Policy object
    eps: float
        probability of selecting a random action (epsilon-greedy policy)
    memorize: boolean 
        whether to store the trajectory in a dict

    Returns
    -------
    TODO
    """

    assert env.name == "RL Training POMDP"

    states_episode = np.empty(env.max_t, dtype=object)
    next_states_episode = np.empty((env.max_t, env.Nactions, env.mod.Nhits), dtype=object)
    
    init_belief, observation, info = env.reset()

    terminated = False
    step = 0
    total_reward = 0
    while terminated == False:

        state, next_states = env.transitions()
        states_episode[step] = state
        next_states_episode[step, :] = next_states

        action = pol.select_epsilon_greedy_action(model, epsilon=eps) 
        observation, terminated, reward, info = env.step(action)
        total_reward += reward
        step += 1

    return states_episode, next_states_episode, total_reward

def run_batch_of_training_episodes(batch_size, env, pol, model, eps):
    """
    Execute batch of episodes to train neural network model.
    NOTE: all episodes in the batch are run with the same exploration rate (eps)
    NOTE: the initial belief state is not stored

    Arguments
    ---------
    batch_size: int
        number of training episodes
    env: TrainingEnv object
    pol: Policy object
    eps: float
        probability of selecting a random action (epsilon-greedy policy)

    Returns
    -------
    states: array of State objects of size (Nsteps * batch_size)
    next_states: array of State objects of size (Nsteps * batch_size, Nactions, Nhits)
    total_rewards: array of the total reward of the episodes (size = batch_size)
    """

    assert env.name == "RL Training POMDP"

    states_batch = np.empty((batch_size, env.max_t), dtype=object)
    next_states_batch = np.empty((batch_size, env.max_t, env.Nactions, env.mod.Nhits), dtype=object)
    total_rewards_batch = np.empty((batch_size,), dtype=float)

    for i_episode in range(batch_size):

        states_episode, next_states_episode, total_reward = run_one_training_episode(env, pol, model, eps)

        states_batch[i_episode, :] = states_episode
        next_states_batch[i_episode, :] = next_states_episode
        total_rewards_batch[i_episode] = total_reward

    states_batch = np.reshape(states_batch, newshape=(batch_size * env.max_t, ))
    next_states_batch = np.reshape(next_states_batch, newshape=(batch_size * env.max_t, env.Nactions, env.mod.Nhits))

    return states_batch, next_states_batch, total_rewards_batch