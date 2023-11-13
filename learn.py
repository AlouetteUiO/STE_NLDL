import cProfile
import pstats

import multiprocessing as mp
from functools import partial
from itertools import repeat, chain
import numpy as np
import scipy
from copy import deepcopy
import os
import time
import sys
sys.path.insert(1, os.path.abspath(os.path.join(sys.path[0], '..', '..')))

import tensorflow as tf
print(f"Numpy version: {np.__version__}")
print(f"Scipy version: {scipy.__version__}")
print(f"Tensorflow version: {tf.__version__}")
print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

from ste_NLDL.classes.config import Config
from ste_NLDL.classes.neuralnetwork import ValueModel, reload_model
from ste_NLDL.functions.run_episode import run_batch_of_training_episodes, run_one_training_episode, run_one_episode
from ste_NLDL.functions.stats import evaluate_policy, plot_stats, plot_training_data_evolution, save_stats

# config
np.set_printoptions(precision=4)
import matplotlib
matplotlib.use('Agg')

# Global parameters
# Use this if you want to continue with a saved model
# Note that parameters for NN in config input file should match the NN model you load
continue_training = False
model_path = "ste_NLDL/learn/models/20230818-100332/20230818-100332_model_bkp_34"

RUN_NAME = time.strftime("%Y%m%d-%H%M%S")
print(f"RUN_NAME = {RUN_NAME}")

DIR_OUTPUTS = os.path.abspath(os.path.join(sys.path[0], "outputs", RUN_NAME))
DIR_MODELS = os.path.abspath(os.path.join(sys.path[0], "models", RUN_NAME))
print(f"DIR_OUTPUTS = {DIR_OUTPUTS}")
print(f"DIR_MODELS = {DIR_MODELS}")

def save_parameters(env, model):
    """
    Save parameters in a file and print them on screen.
    TODO This is not in yml format yet!!! Fix this!
    """
    param_file = os.path.join(DIR_OUTPUTS, str(RUN_NAME + "_parameters" + ".yml"))
    pfile = open(param_file, "a")
    for out in (None, pfile):
        print("Tensorflow version = " + str(tf.__version__), file=out)
        
        print("* Problem parameters", file=out)
        print("N_DIMS = " + str(env.mod.Ndim), file=out)
        print("V = " + str(env.V), file=out)
        print("D = " + str(env.D), file=out)
        print("N_ACTIONS = " + str(env.Nactions), file=out)
        print("N_GRID = " + str(env.Ngrid), file=out)
        print("N_HITS = " + str(env.mod.Nhits), file=out)
        print("STOP_t = " + str(STOP_t), file=out)
        print("BELIEF_CRITERIA = " + str(env.belief_criteria), file=out)

        print("* Parallelization", file=out)
        print("N_PARALLEL = 1", file=out)  # NOTE parallel does not work yet!
        # print("N_PARALLEL = " + str(N_PARALLEL), file=out) # nothing run in parallel at the moment

        print("* Exploration", file=out)
        print("E_GREEDY_FLOOR = " + str(E_GREEDY_FLOOR), file=out)
        print("E_GREEDY_0 = " + str(E_GREEDY_0), file=out)
        print("E_GREEDY_DECAY = " + str(E_GREEDY_DECAY), file=out)
        print("RANDOM_EPISODES = " + str(RANDOM_EPISODES), file=out)

        print("* Gradient descent", file=out)
        print("BATCH_SIZE = " + str(BATCH_SIZE), file=out)
        print("N_GD_STEPS = " + str(N_GD_STEPS), file=out)
        print("LEARNING_RATE = " + str(LEARNING_RATE), file=out)

        print("* Experience replay", file=out)
        print("MEMORY_SIZE = " + str(MEMORY_SIZE), file=out)
        print("NEW_TRANS_PER_IT = " + str(NEW_TRANS_PER_IT), file=out)
        print("REPLAY_NTIMES = " + str(REPLAY_NTIMES), file=out)

        print("* Other DQN parameters", file=out)
        print("ALGO_MAX_IT = " + str(ALGO_MAX_IT), file=out)
        print("UPDATE_FROZEN_MODEL_EVERY = " + str(UPDATE_FROZEN_MODEL_EVERY), file=out)

        print("* NN architecture", file=out)
        print("FC_LAYERS = " + str(FC_LAYERS), file=out)
        print("FC_UNITS = " + str(FC_UNITS), file=out)
        print("CNN_LAYERS = " + str(CNN_LAYERS), file=out)
        print("CNN_FILTERS = " + str(CNN_FILTERS), file=out)
        print("CNN_KERNEL_SIZE = " + str(CNN_KERNEL_SIZE), file=out)
        print("POOLING = " + str(POOLING), file=out)
        print("POOLING_KERNEL_SIZE = " + str(POOLING_KERNEL_SIZE), file=out)
        print("POOLING_STRIDES = " + str(POOLING_STRIDES), file=out)
        print("REG_FACTOR = " + str(REG_FACTOR), file=out)
        Nweights = np.sum([np.prod(w.shape) for w in model.get_weights()])
        print("Number of weights = ", Nweights, file=out)
        print("input shape = ", env.NN_input_shape, file=out)
        if out is None:
            model.summary()
        else:
            model.summary(print_fn=lambda x: out.write(x + '\n'))

        print("* Performance evaluation", file=out)
        print("EVALUATE_PERFORMANCE_EVERY = " + str(EVALUATE_PERFORMANCE_EVERY), file=out)
        print("EVALUATE_PERFORMANCE_STATS_EVERY = " + str(EVALUATE_PERFORMANCE_STATS_EVERY), file=out)
        print("N_RUNS_STATS = " + str(N_RUNS_STATS), file=out)

        print("* Save parameters", file=out)
        print("SAVE_MODEL_EVERY = " + str(SAVE_MODEL_EVERY), file=out)
        print("SAVE_MODEL_BACKUP_EVERY = " + str(EVALUATE_PERFORMANCE_EVERY), file=out)

    pfile.close()
    sys.stdout.flush()

def init_(cfg, ref):
    """ initiate the environment """

    environment_index = cfg.get("methods/environment")
    policy_index = cfg.get("methods/policy")
    observationmodel_index = cfg.get("methods/observationmodel")
    bayes_index = cfg.get("methods/bayes")

    if environment_index == -1:
        from ste_NLDL.classes.training import TrainingEnv as Environment
        # TODO We got to change this. Make it more clear that TrainingEnv is part the Environment for DRL
    if observationmodel_index == 2:
        from ste_NLDL.classes.observationmodel import VergassolaPlumeModel as ObservationModel
    if bayes_index == 0:
        from ste_NLDL.classes.bayes import AnalyticalBayes as Bayes
    if policy_index == -1:
        from ste_NLDL.classes.policy import DRLpolicy as Policy
    from ste_NLDL.classes.neuralnetwork import ValueModel

    if ref:
        from ste_NLDL.classes.environment import POMDP as RefEnvironment
        from ste_NLDL.classes.policy import Infotaxis as RefPolicy

    mod = ObservationModel(cfg)
    print(f"model: {mod.index, mod.name}")
    bay = Bayes(cfg, mod)
    print(f"bayes: {bay.index, bay.name}")
    env = Environment(cfg, mod, bay)
    print(f"environment: {env.index, env.name}")
    pol = Policy(cfg, env) # if policy is infotaxis then input is cfg, env, bay
    print(f"policy: {pol.index, pol.name}")

    if ref:
        refenv = RefEnvironment(cfg, mod, bay)
        refpol = RefPolicy(cfg, refenv, bay)
    else:
        refenv = None
        refpol = None

    return env, pol, refenv, refpol, mod

def build_model(
    model_name, 
    Ndim,
    FC_layers,
    FC_units,
    CNN_layers,
    CNN_filters,
    CNN_kernel_size,
    pooling,
    pooling_kernel_size,
    pooling_strides,
    reg_factor,
    learning_rate,
    ):
    """
    Compile and build the neural network model.

    Parameters
    ----------
    model_name: string
        'onlinie_network' or 'target_network'
    Ndim: int
        number of space dimensions for the search problem
    FC_layers: int
        number of hidden layers
    FC_units: int or tuple
        units per layer
    learning_rate: float
        usual learning rate 

    Returns
    -------
    model: build and compiled ValueModel object
        instance of the neural network model
    """

    model = ValueModel(
        model_name = model_name,
        Ndim = Ndim,
        FC_layers = FC_layers,
        FC_units = FC_units,
        CNN_layers = CNN_layers,
        CNN_filters = CNN_filters,
        CNN_kernel_size = CNN_kernel_size,
        pooling = pooling,
        pooling_kernel_size = pooling_kernel_size,
        pooling_strides = pooling_strides,
        regularization_factor = reg_factor,
    )

    model.build_graph(belief_map_shape_nobatch = env.NN_input_shape)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)  
    model.compile(optimizer=optimizer)

    return model

def run():
    """
    Main function
    """
    print("\n*** Save parameters. This is a summary: ")
    save_parameters(env, my_model)

    # Compute stats of reference infotaxis policy
    if ref:
        print("\n*** Compute stats of the infotaxis reference policy")
        filename = os.path.join(DIR_OUTPUTS, str(RUN_NAME + "_ref"))
        ref_stats = evaluate_policy(filename, N_RUNS_STATS, refenv, refpol, model=None, print_stats_bool=True)
        filename_txt = os.path.join(DIR_OUTPUTS, str(RUN_NAME + "_stats_ref.txt"))
        timestamp = time.perf_counter()
        save_stats('ref', timestamp-timestamp_start, ref_stats, filename_txt) 

    print("\n*** Learn optimal value function")
    train_model(env, pol)
    print("---- DONE with training ----")
    
    print("\n*** Compute final stats of the RL policy")
    filename = os.path.join(DIR_OUTPUTS, str(RUN_NAME + "_RL_final"))
    stats = evaluate_policy(filename, N_RUNS_STATS, env, pol, model=my_model, print_stats_bool=True) 
    title = f"evaluation of final RL policy {pol}"
    plot_stats(filename, title)

def run_one_training_episode_parallel_wrapper(model_path, eps, i):
    with tf.compat.v1.Session() as sess:
        #my_experiencemodel = reload_model(model_path, inputshape = env.NN_input_shape) 
        states_episode, next_states_episode, total_reward = run_one_training_episode(env, pol, my_model, eps) #my_experiencemodel, eps)
        #del my_experiencemodel  # TODO does this help reduce memory?
        tf.keras.backend.clear_session()
    return states_episode, next_states_episode, total_reward

def combine_npy_files(num_sim, data_shape, output_shape, data_type, file_string):
    """Combine all data files in the directory temporary to one and delete them."""
    solution = np.empty((num_sim, *data_shape), dtype=data_type) # for states and next_states
    path = "temporary/"
    for idx in range(num_sim):
        curr_file = path + file_string + "_" + str(idx) + ".npy"
        data = np.load(curr_file, allow_pickle=True)
        solution[idx, :] = data
        os.remove(curr_file)
    solution = np.reshape(solution, newshape=output_shape) 
    np.save("temporary/" + file_string + ".npy", solution)

### Training core algorithm ###

def train_model(env, pol):
    """
    Train the model to an (approximately) optimal value function using a model-based version of DQN.
    """

    def get_epsilon_exploration(iteration, random_episodes, epsilon_decay, epsilon_floor):
        """
        Compute the epsilon (exploration factor) for epsilon-greedy exploration 
        during training. A random action is selected with epsilon probability.

        Arguments
        ---------
        iteration: int
            current iteration
        random_episodes: int
            number of random episodes at start of training phase
        epsilon_decay: int
            decay timescale for the exploration parameter epsilon
        epsilon_floor: float
            floor value of the exploration parameter epsilon

        Returns
        -------
        epsilon: float
            exploration parameter
        """
        if iteration < random_episodes:
            return 1.0
        else:
            if epsilon_decay == None:
                return epsilon_floor
            else:
                return max(np.exp(-(iteration-random_episodes) / epsilon_decay), epsilon_floor)
        
    def get_target(next_states):
        """
        Compute the target value (for training) of a state s.
        This is the estimated optimal value for state s.

        The frozen model is used for the prediction of future rewards. 
        The frozen model is also called the target network.

        Arguments
        ---------
        next_states: array of State objects with shape (Nactions, Nhits) or (batch_size, Nactions, Nhits)

        Returns
        -------
        target: array of target values with shape (batch_size, 1)
            target values for all next states
        """
        target = env.get_target(model=my_frozenmodel, next_states=next_states)
        return target 
    
    def get_experience(batch_size, eps, env, pol, model, parallel):
        """
        Generate new experiences 
        """

        assert model.model_name == 'online_network'

        if parallel:
            raise ValueError("get_experience is not yet implmented correctly in parallel")
        else:
            states, next_states, total_rewards = run_batch_of_training_episodes(batch_size=batch_size, 
                                                                                env=env, 
                                                                                pol=pol, 
                                                                                model=my_model,
                                                                                eps=eps)

        assert states.shape[0] == next_states.shape[0]
        #print("cut-off states if it is too long to be stored in memory")
        if states.shape[0] > MEMORY_SIZE:
            states = states[:MEMORY_SIZE]
            next_states = next_states[:MEMORY_SIZE]
        return states, next_states, total_rewards
    
    def evaluate_performance(it, N_transitions_seen, parallel):
        """
        Evaluate performance of model
        Then compute the stats
        """
        if parallel:
            raise ValueError("evaluate_performance is not yet implmented in parallel")
        else:
            filename = os.path.join(DIR_OUTPUTS, str(RUN_NAME + "_RL_" + str(it // EVALUATE_PERFORMANCE_STATS_EVERY)))
            stats = evaluate_policy(filename, N_RUNS_STATS, env, pol, model=my_model, print_stats_bool=True)
        
        title = f"evaluation of current RL policy {pol}, training it = {it}, transitions seen = {N_transitions_seen}"
        plot_stats(filename, title)
        filename_txt = os.path.join(DIR_OUTPUTS, str(RUN_NAME + "_stats_NN.txt"))
        timestamp = time.perf_counter()
        save_stats(it, timestamp-timestamp_start, stats, filename_txt)
        return None

    print("* populating memory...")
    Nepisodes = max(int(0.9 * MEMORY_SIZE / STOP_t), 1)
    print("Nepisodes: ", Nepisodes)
    eps = E_GREEDY_0
    print("eps: ", eps)
    parallel = Nepisodes >= N_PARALLEL > 1
    #print(f"parallel = {parallel}")
    print(f"get experience over {Nepisodes} episodes to populate memory")
    states, next_states, total_rewards = get_experience(batch_size=Nepisodes, eps=eps, env=env, pol=pol, model=my_model, parallel=False)
    print("occupied memory: ", states.shape[0], "/", MEMORY_SIZE)

    print(my_model.summary())
    for layer in my_model.layers:
        print(layer)

    print("* start training...")
    it = 0
    # for stats
    data = np.nan * np.zeros(4)
    N_transitions_generated = states.shape[0]
    N_transitions_seen = 0

    while it <= ALGO_MAX_IT: # max_it = ALGO_MAX_IT (= 10000 for zoo test)

        # *** save model
        if it % SAVE_MODEL_EVERY == 0:
            model_name = str(RUN_NAME + '_model')
            model_path = os.path.join(DIR_MODELS, model_name)
            my_model.save_model(model_path)

        # *** evaluate performance
        if it % EVALUATE_PERFORMANCE_STATS_EVERY == 0:
            print(f"* evaluating performance of the current model-based policy at iteration {it} ...")
            # save model
            model_name = str(RUN_NAME + '_model_bkp_' + str(it // EVALUATE_PERFORMANCE_STATS_EVERY))
            model_path = os.path.join(DIR_MODELS, model_name)
            my_model.save_model(model_path)
            # compute, print and plot performance statistics
            evaluate_performance(it, N_transitions_seen, parallel=False)
            print("* done with evaluating the current model-based policy")

        # *** Generate experience
        Nepisodes = max(int(np.ceil(NEW_TRANS_PER_IT / STOP_t)), 1)
        eps = get_epsilon_exploration(it, random_episodes=RANDOM_EPISODES, epsilon_decay=E_GREEDY_DECAY, epsilon_floor=E_GREEDY_FLOOR)
        parallel = Nepisodes >= N_PARALLEL > 1
        new_states, new_next_states, total_rewards = get_experience(batch_size=Nepisodes, eps=eps, env=env, pol=pol, model=my_model, parallel=False)

        # *** delete the oldest experiences and add the new ones to memory
        states = update_buffer_memory(states, new_states, max_size=MEMORY_SIZE)
        next_states = update_buffer_memory(next_states, new_next_states, max_size=MEMORY_SIZE)
        renewed_mem = new_states.shape[0] / MEMORY_SIZE # fraction of memory that is updated by running the new episodes
        N_transitions_generated += new_states.shape[0]

        ### Training ###

        # *** update model used for computing targets
        if it % UPDATE_FROZEN_MODEL_EVERY == 0:
            my_frozenmodel.set_weights(my_model.get_weights())

        # *** perform step(s) of mini-batch gradient descent
        mean_loss = 0
        memory_size = states.shape[0]
        for gd_step in range(N_GD_STEPS):
            batch = np.random.randint(memory_size, size=BATCH_SIZE) # randomly select a batch of data from memory
            x, _, _ = env.states2inputs(states[batch])
            y = get_target(next_states[batch])
            loss = my_model.train_step(x, y) # train (model weights are updated) # NOTE training = True
            mean_loss += loss
        mean_loss /= N_GD_STEPS # mean loss over this mini-batch training iteration 

        N_transitions_seen += N_GD_STEPS * BATCH_SIZE
        it += 1

        # *** Save training info
        add_data = np.array([it, eps, mean_loss, np.mean(total_rewards)])
        data = np.vstack((data, add_data))
        if it % EVALUATE_PERFORMANCE_EVERY == 0:
            print("Plot training data evaluation... ")
            filename = os.path.join(DIR_OUTPUTS, str(RUN_NAME + "_figure_training_stats.png"))
            title = f"evolution of performance during training (currently: training it = {str(it)}, transitions seen = {N_transitions_seen})"
            plot_training_data_evolution(data, filename, title)

        # *** Print info to screen
        if it % 1 == 0:

            print(
                "---- training iteration:", it,
                "  |  eps:", eps,
                "  |  episodes added:", Nepisodes,
                "  |  fraction memory renewed:", renewed_mem,
                "  |  transitions seen:", N_transitions_seen,
                "  |  loss:", mean_loss,
                "  |  ", time.strftime("%Y-%m-%d %H:%M:%S")
            )

    print("Stopped: max number of training iterations reached.")
    return None


# FUNCTIONS USED BY TRAIN_MODEL --------------------

def update_buffer_memory(mem, new, max_size):
    """
    Add new transitions to the memory buffer.

    Args:
        mem (ndarray): current memory buffer
        new (ndarray): array of new transitions
        max_size (int): max number of transitions in memory

    Returns:
        mem (ndarray): updated memory
    """
    mem_size = mem.shape[0]
    new_size = new.shape[0]
    if new_size > max_size:
        raise Exception("Memory is too small to add this many new episodes!")
    if mem_size + new_size <= max_size:
        mem = np.concatenate((mem, new), axis=0)
    else:
        delete_size = mem_size + new_size - max_size
        mem = np.concatenate((mem[delete_size:], new), axis=0)

    assert mem.shape[0] <= max_size

    return mem

# --------------------------------------------------

if __name__ == '__main__':

    timestamp_start = time.perf_counter()

    if not os.path.isdir(DIR_OUTPUTS):
        os.makedirs(DIR_OUTPUTS)

    if not os.path.isdir(DIR_MODELS):
        os.makedirs(DIR_MODELS)

    ref = True

    # autoset parameters here
    cfg = Config("test_ste/model_test_2D_5.yml")
    env, pol, refenv, refpol, mod = init_(cfg, ref) 

    # Load variables
    SAVE_MODEL_EVERY = int(cfg.get("nn/save_model_every"))
    N_RUNS_STATS = int(cfg.get("nn/n_runs_stats"))
    EVALUATE_PERFORMANCE_EVERY = int(cfg.get("nn/evaluate_performance_every"))
    EVALUATE_PERFORMANCE_STATS_EVERY = int(cfg.get("nn/evaluate_performance_stats_every"))
    FC_LAYERS = int(cfg.get("nn/fc_layers"))
    FC_UNITS = cfg.get("nn/fc_units")
    CNN_LAYERS = int(cfg.get("nn/cnn_layers"))
    CNN_FILTERS = cfg.get("nn/cnn_filters")
    CNN_KERNEL_SIZE = cfg.get("nn/cnn_kernel_size")
    REG_FACTOR = cfg.get("nn/reg_factor")
    POOLING = cfg.get("nn/pooling")
    POOLING_KERNEL_SIZE = cfg.get("nn/pooling_kernel_size")
    POOLING_STRIDES = cfg.get("nn/pooling_strides")
    UPDATE_FROZEN_MODEL_EVERY = int(cfg.get("nn/update_frozen_model_every"))
    ALGO_MAX_IT = int(cfg.get("nn/algo_max_it"))
    LEARNING_RATE = cfg.get("nn/learning_rate")
    STOP_t = int(cfg.get("nn/stop_t"))
    N_PARALLEL = int(cfg.get("nn/n_parallel"))
    E_GREEDY_FLOOR = cfg.get("nn/e_greedy_floor") 
    E_GREEDY_0 = cfg.get("nn/e_greedy_0")
    RANDOM_EPISODES = int(cfg.get("nn/random_episodes"))
    E_GREEDY_DECAY = int(cfg.get("nn/e_greedy_decay")) # decay timescale of eps in number of training iterations
    MAX_IT = int(cfg.get("nn/algo_max_it"))
    MEMORY_SIZE = int(cfg.get("nn/memory_size")) # number of transitions (s, s') to keep in memory
    BATCH_SIZE = int(cfg.get("nn/batch_size"))  # size of the mini-batch
    N_GD_STEPS = int(cfg.get("nn/n_gd_steps"))  # number of gradient descent steps per training iteration
    REPLAY_NTIMES = int(cfg.get("nn/replay_ntimes"))  # how many times a transition is used for training before being deleted, on average
    NEW_TRANS_PER_IT = int(BATCH_SIZE * N_GD_STEPS / REPLAY_NTIMES)
    if NEW_TRANS_PER_IT > 0.8 * MEMORY_SIZE:
        print("Nb of new transitions per it (approx): ", NEW_TRANS_PER_IT)
        print("Memory size:", MEMORY_SIZE)
        raise Exception("Memory is too small for these BATCH_SIZE, N_GD_STEPS and REPLAY_NTIMES")
    elif NEW_TRANS_PER_IT < 1:
        raise Exception("Not enough new transitions per it, increase BATCH_SIZE or N_GD_STEPS")

    # model initialization
    print("\n*** Building models...")

    my_model = build_model(
        model_name = 'online_network',
        Ndim = env.Ndim,
        FC_layers = FC_LAYERS,
        FC_units = FC_UNITS,
        CNN_layers = CNN_LAYERS,
        CNN_filters = CNN_FILTERS,
        CNN_kernel_size = CNN_KERNEL_SIZE,
        pooling = POOLING,
        pooling_kernel_size = POOLING_KERNEL_SIZE,
        pooling_strides = POOLING_STRIDES,
        reg_factor = REG_FACTOR,
        learning_rate = LEARNING_RATE,
    )

    my_frozenmodel = build_model(
        model_name = 'target_network',
        Ndim = env.Ndim,
        FC_layers = FC_LAYERS,
        FC_units = FC_UNITS,
        CNN_layers = CNN_LAYERS,
        CNN_filters = CNN_FILTERS,
        CNN_kernel_size = CNN_KERNEL_SIZE,
        pooling = POOLING,
        pooling_kernel_size = POOLING_KERNEL_SIZE,
        pooling_strides = POOLING_STRIDES,
        reg_factor = REG_FACTOR,
        learning_rate = LEARNING_RATE,
    )
    # set weights on frozenmodel
    my_frozenmodel.set_weights(my_model.get_weights())

    if continue_training:
        print(f"loading previous weights from: {model_path}")
        my_old_model = reload_model(model_path, inputshape = env.NN_input_shape)
        my_model.set_weights(my_old_model.get_weights())
        my_frozenmodel.set_weights(my_old_model.get_weights())
        del my_old_model

    # Main program
    #cProfile.run("run()", filename="profile_results.prof")
    #profile_results = pstats.Stats("profile_results.prof")
    #profile_results.strip_dirs().sort_stats("cumulative")
    #profile_results.print_stats(150)
    run() 

    timestamp_end = time.perf_counter()
    print(f"wall clock time = {timestamp_end - timestamp_start}")