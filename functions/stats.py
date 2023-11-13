"""
Functions to evaluate the performance of policies
"""

import numpy as np
import yaml
import os
import matplotlib.pyplot as plt
from ste.functions.run_episode import run_batch_of_evaluation_episodes


def compute_stats(filename):
    """
    Compute the statistics of the evaluated episodes
    """

    reward_batch = np.load(filename + "_reward.npy", allow_pickle=True)
    converged_batch = np.load(filename + "_converged.npy", allow_pickle=True)
    success_source_batch = np.load(filename + "_success_source.npy", allow_pickle=True)
    success_flux_batch = np.load(filename + "_success_flux.npy", allow_pickle=True)
    DRPS_x_batch = np.load(filename + "_DRPS_x.npy", allow_pickle=True)
    DRPS_y_batch = np.load(filename + "_DRPS_y.npy", allow_pickle=True)
    DRPS_q_batch = np.load(filename + "_DRPS_q.npy", allow_pickle=True)
    DRPS_xrel_batch = np.load(filename + "_DRPS_xrel.npy", allow_pickle=True)
    DRPS_yrel_batch = np.load(filename + "_DRPS_yrel.npy", allow_pickle=True)
    DRPS_qrel_batch = np.load(filename + "_DRPS_qrel.npy", allow_pickle=True)

    batch_size = reward_batch.shape[0]

    success_source_and_flux = np.logical_and(success_source_batch, success_flux_batch)
    success_source_and_flux_rate = np.sum(success_source_and_flux, axis=0) / batch_size
    success_source_rate = np.sum(success_source_batch, axis=0) / batch_size  
    success_flux_rate = np.sum(success_flux_batch, axis=0) / batch_size  
    converged_rate = np.sum(converged_batch, axis=0) / batch_size

    success_only_source_rate = success_source_rate - success_source_and_flux_rate  # flux incorrect
    success_only_flux_rate = success_flux_rate - success_source_and_flux_rate  # source incorrect
    converged_wrong_rate = converged_rate - success_only_source_rate - success_only_flux_rate - success_source_and_flux_rate
    not_converged_rate = 1 - converged_rate

    total_rewards_batch = np.sum(reward_batch, axis=1)
    total_rewards_mean = np.mean(total_rewards_batch)
    total_rewards_std = np.std(total_rewards_batch)
    total_rewards_median = np.median(total_rewards_batch)
    total_rewards_5percentile = np.percentile(total_rewards_batch, q=5)
    total_rewards_25percentile = np.percentile(total_rewards_batch, q=25)
    total_rewards_75percentile = np.percentile(total_rewards_batch, q=75)
    total_rewards_95percentile = np.percentile(total_rewards_batch, q=95)

    rewards_mean = np.mean(reward_batch, axis=0)
    rewards_std = np.std(reward_batch, axis=0)
    rewards_median = np.median(reward_batch, axis=0)
    rewards_5percentile  = np.percentile(reward_batch, q=5, axis=0)
    rewards_25percentile = np.percentile(reward_batch, q=25, axis=0)
    rewards_75percentile = np.percentile(reward_batch, q=75, axis=0)
    rewards_95percentile = np.percentile(reward_batch, q=95, axis=0)

    DRPS_mean_flux = np.mean(DRPS_q_batch)
    DRPS_mean_x = np.mean(DRPS_x_batch)
    DRPS_mean_y = np.mean(DRPS_y_batch)

    DRPS_std_flux = np.std(DRPS_q_batch)
    DRPS_std_x = np.std(DRPS_x_batch)
    DRPS_std_y = np.std(DRPS_y_batch)

    DRPS_median_flux = np.median(DRPS_q_batch)
    DRPS_median_x = np.median(DRPS_x_batch)
    DRPS_median_y = np.median(DRPS_y_batch)
    DRPS_median_flux_rel = np.median(DRPS_qrel_batch)
    DRPS_median_x_rel = np.median(DRPS_xrel_batch)  
    DRPS_median_y_rel = np.median(DRPS_yrel_batch)

    DRPS_5percentile_flux = np.percentile(DRPS_q_batch, q=5)
    DRPS_5percentile_x = np.percentile(DRPS_x_batch, q=5)
    DRPS_5percentile_y = np.percentile(DRPS_y_batch, q=5)

    DRPS_25percentile_flux = np.percentile(DRPS_q_batch, q=25)
    DRPS_25percentile_x = np.percentile(DRPS_x_batch, q=25)
    DRPS_25percentile_y = np.percentile(DRPS_y_batch, q=25)

    DRPS_75percentile_flux = np.percentile(DRPS_q_batch, q=75)
    DRPS_75percentile_x = np.percentile(DRPS_x_batch, q=75)
    DRPS_75percentile_y = np.percentile(DRPS_y_batch, q=75)
    DRPS_75percentile_flux_rel = np.percentile(DRPS_qrel_batch, q=75)
    DRPS_75percentile_x_rel = np.percentile(DRPS_xrel_batch, q=75)
    DRPS_75percentile_y_rel = np.percentile(DRPS_yrel_batch, q=75)

    DRPS_95percentile_flux = np.percentile(DRPS_q_batch, q=95)
    DRPS_95percentile_x = np.percentile(DRPS_x_batch, q=95)
    DRPS_95percentile_y = np.percentile(DRPS_y_batch, q=95)
    DRPS_95percentile_flux_rel = np.percentile(DRPS_qrel_batch, q=95)
    DRPS_95percentile_x_rel = np.percentile(DRPS_xrel_batch, q=95)
    DRPS_95percentile_y_rel = np.percentile(DRPS_yrel_batch, q=95)

    np.savez(filename + ".npz",
        total_rewards_mean = total_rewards_mean,
        total_rewards_std = total_rewards_std,
        total_rewards_median = total_rewards_median,
        total_rewards_5percentile = total_rewards_5percentile,
        total_rewards_25percentile = total_rewards_25percentile,
        total_rewards_75percentile = total_rewards_75percentile,
        total_rewards_95percentile = total_rewards_95percentile,
        rewards_mean = rewards_mean,
        rewards_std = rewards_std,
        rewards_median = rewards_median,
        rewards_5percentile = rewards_5percentile,
        rewards_25percentile = rewards_25percentile,
        rewards_75percentile = rewards_75percentile,
        rewards_95percentile = rewards_95percentile,
        success_source_and_flux_rate = success_source_and_flux_rate,
        success_source_rate = success_source_rate,
        success_flux_rate = success_flux_rate,
        converged_rate = converged_rate,
        success_only_source_rate = success_only_source_rate,
        success_only_flux_rate = success_only_flux_rate,
        converged_wrong_rate = converged_wrong_rate,
        not_converged_rate = not_converged_rate,
        DRPS_mean_flux = DRPS_mean_flux,
        DRPS_mean_x = DRPS_mean_x,
        DRPS_mean_y = DRPS_mean_y,
        DRPS_std_flux = DRPS_std_flux,
        DRPS_std_x = DRPS_std_x,
        DRPS_std_y = DRPS_std_y,
        DRPS_median_flux = DRPS_median_flux,
        DRPS_median_x = DRPS_median_x,
        DRPS_median_y = DRPS_median_y,
        DRPS_5percentile_flux = DRPS_5percentile_flux,
        DRPS_5percentile_x = DRPS_5percentile_x,
        DRPS_5percentile_y = DRPS_5percentile_y,
        DRPS_25percentile_flux = DRPS_25percentile_flux,
        DRPS_25percentile_x = DRPS_25percentile_x,
        DRPS_25percentile_y = DRPS_25percentile_y,
        DRPS_75percentile_flux = DRPS_75percentile_flux,
        DRPS_75percentile_x = DRPS_75percentile_x,
        DRPS_75percentile_y = DRPS_75percentile_y,
        DRPS_95percentile_flux = DRPS_95percentile_flux,
        DRPS_95percentile_x = DRPS_95percentile_x,
        DRPS_95percentile_y = DRPS_95percentile_y,
        DRPS_rel_median_flux = DRPS_median_flux_rel,
        DRPS_rel_median_x = DRPS_median_x_rel,
        DRPS_rel_median_y = DRPS_median_y_rel,
        DRPS_rel_75percentile_flux = DRPS_75percentile_flux_rel,
        DRPS_rel_75percentile_x = DRPS_75percentile_x_rel,
        DRPS_rel_75percentile_y = DRPS_75percentile_y_rel,
        DRPS_rel_95percentile_flux = DRPS_95percentile_flux_rel,
        DRPS_rel_95percentile_x = DRPS_95percentile_x_rel,
        DRPS_rel_95percentile_y = DRPS_95percentile_y_rel,
    )

    os.remove(filename + "_reward.npy")
    os.remove(filename + "_converged.npy")
    os.remove(filename + "_success_source.npy")
    os.remove(filename + "_success_flux.npy")
    os.remove(filename + "_DRPS_x.npy")
    os.remove(filename + "_DRPS_y.npy")
    os.remove(filename + "_DRPS_q.npy")
    os.remove(filename + "_DRPS_xrel.npy")
    os.remove(filename + "_DRPS_yrel.npy")
    os.remove(filename + "_DRPS_qrel.npy")
    
    # create dict
    stats = {
        "total_rewards_mean": total_rewards_mean,
        "total_rewards_std": total_rewards_std,
        "total_rewards_median": total_rewards_median,
        "total_rewards_5percentile": total_rewards_5percentile,
        "total_rewards_25percentile": total_rewards_25percentile,
        "total_rewards_75percentile": total_rewards_75percentile,
        "total_rewards_95percentile": total_rewards_95percentile,
        "rewards_mean": rewards_mean,
        "rewards_std": rewards_std,
        "rewards_median": rewards_median,
        "rewards_5percentile": rewards_5percentile,
        "rewards_25percentile": rewards_25percentile,
        "rewards_75percentile": rewards_75percentile,
        "rewards_95percentile": rewards_95percentile,
        "success_source_and_flux_rate": success_source_and_flux_rate,
        "success_source_rate": success_source_rate,
        "success_flux_rate": success_flux_rate,
        "converged_rate": converged_rate,
        "success_only_flux_rate": success_only_flux_rate,
        "success_only_source_rate": success_only_source_rate,
        "converged_wrong_rate": converged_wrong_rate,
        "not_converged_rate": not_converged_rate,
        "DRPS_mean_flux": DRPS_mean_flux,
        "DRPS_mean_x": DRPS_mean_x,
        "DRPS_mean_y": DRPS_mean_y,
        "DRPS_std_flux": DRPS_std_flux,
        "DRPS_std_x": DRPS_std_x,
        "DRPS_std_y": DRPS_std_y,
        "DRPS_median_flux": DRPS_median_flux,
        "DRPS_median_x": DRPS_median_x,
        "DRPS_median_y": DRPS_median_y,
        "DRPS_5percentile_flux": DRPS_5percentile_flux,
        "DRPS_5percentile_x": DRPS_5percentile_x,
        "DRPS_5percentile_y": DRPS_5percentile_y,
        "DRPS_25percentile_flux": DRPS_25percentile_flux,
        "DRPS_25percentile_x": DRPS_25percentile_x,
        "DRPS_25percentile_y": DRPS_25percentile_y,
        "DRPS_75percentile_flux": DRPS_75percentile_flux,
        "DRPS_75percentile_x": DRPS_75percentile_x,
        "DRPS_75percentile_y": DRPS_75percentile_y,
        "DRPS_95percentile_flux": DRPS_95percentile_flux,
        "DRPS_95percentile_x": DRPS_95percentile_x,
        "DRPS_95percentile_y": DRPS_95percentile_y,
        "DRPS_rel_median_flux": DRPS_median_flux_rel,
        "DRPS_rel_median_x": DRPS_median_x_rel,
        "DRPS_rel_median_y": DRPS_median_y_rel,
        "DRPS_rel_75percentile_flux": DRPS_75percentile_flux_rel,
        "DRPS_rel_75percentile_x": DRPS_75percentile_x_rel,
        "DRPS_rel_75percentile_y": DRPS_75percentile_y_rel,
        "DRPS_rel_95percentile_flux": DRPS_95percentile_flux_rel,
        "DRPS_rel_95percentile_x": DRPS_95percentile_x_rel,
        "DRPS_rel_95percentile_y": DRPS_95percentile_y_rel,
    }

    return stats

def evaluate_policy(filename, Nepisodes, env, pol, model, print_stats_bool=False):
    """
    Runs a number of evaluation episodes and computes performance statistics of a given policy.

    Arguments
    ---------
    filename: str
    Nepisodes: int
        number of episodes to compute the statistics over
    env: Environment class
    pol: Policy class
    max_steps: int
        maximum number of steps the agent can take before being timed-out

    Returns
    -------
    stats: dict
    """

    if model:
        assert model.model_name == 'online_network'

    run_batch_of_evaluation_episodes(filename, Nepisodes, env, pol, model)
    print(f"ran batch of {Nepisodes} evaluation episodes")

    stats = compute_stats(filename)

    if print_stats_bool:
        print_stats(stats)

    return stats


def print_stats(stats):
    """
    Prints stats on screen

    Arguments
    ---------
    stats: dict
        dictionary containing statistics computed by function compute_stats
    """
    total_reward = stats["total_rewards_mean"]
    success_source_and_flux = stats['success_source_and_flux_rate'][-1] * 100
    success_source = stats['success_source_rate'][-1] * 100
    success_flux = stats['success_flux_rate'][-1] * 100
    converged = stats['converged_rate'][-1] * 100

    print("--- evaluate model ---")  # can be valuemodel or infotaxis
    print(f"total reward = {total_reward}")
    print(f"converged = {converged} %")
    print(f"found correct source and flux = {success_source_and_flux} %")
    print(f"found correct source = {success_source} %")
    print(f"found correct flux = {success_flux} %")


def plot_stats(filename, title):
    """
    Plot performance stats of evaluation of current value model in a figure and save it.
    """
    # Load data
    with np.load(filename + ".npz") as data:
        rewards_mean = data["rewards_mean"]
        rewards_std = data["rewards_std"]
        data_dict = {
            "NN: correct source and flux": data["success_source_and_flux_rate"],
            "NN: correct source and incorrect flux": data["success_only_source_rate"], 
            "NN: incorrect source and correct flux": data["success_only_flux_rate"],
            "NN: incorrect source and incorrect flux": data["converged_wrong_rate"], 
            "NN: not converged": data["not_converged_rate"],
        }

    filename_base = filename.split("_")[:-2]
    filename_ref = "_".join(filename_base) + "_ref.npz"
    try:
        with np.load(filename_ref) as ref_data:
            ref_stats = True
            rewards_ref_mean = ref_data["rewards_mean"]
            rewards_ref_std = ref_data["rewards_std"]
            ref_data_dict = {
                "infotaxis: correct source and flux": ref_data["success_source_and_flux_rate"],
                "infotaxis: correct source and incorrect flux": ref_data["success_only_source_rate"], 
                "infotaxis: incorrect source and correct flux": ref_data["success_only_flux_rate"],
                "infotaxis: incorrect source and incorrect flux": ref_data["converged_wrong_rate"], 
                "infotaxis: not converged": ref_data["not_converged_rate"],
            }
    except:
        pass

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    plt.subplots_adjust(
        left=0.05, bottom=0.24, right=0.99, top=0.9, wspace=0.15, hspace=0
    )
    colors_NN = ['#08519c', '#3182bd', '#6baed6', '#bdd7e7', '#eff3ff']
    colors_inf = ['#a63603', '#e6550d', '#fd8d3c', '#fdbe85', '#feedde']
    kwargs0 = {"markersize": 5, "linewidth": 2, "color": colors_NN[0]}
    kwargs1 = {"markersize": 5, "linewidth": 2, "color": colors_inf[0]}
    patterns = ["", "", "||", "", ""]

    x = np.arange(1, len(rewards_mean)+1)
    (line,) = ax[0].plot(x, rewards_mean, "-o", label='NN', **kwargs0)
    ax[0].fill_between(x, rewards_mean-rewards_std, rewards_mean+rewards_std, alpha=0.3)
    ax[0].set_title("entropy at number of steps")
    ax[0].set_xlabel("number of steps")
    ax[0].set_ylabel("entropy [nats]") 
    ax[0].set_ylim(bottom=0)
    if ref_stats:
        (line,) = ax[0].plot(x, rewards_ref_mean, "-o", label='infotaxis', **kwargs1)
        ax[0].fill_between(x, rewards_ref_mean-rewards_ref_std, rewards_ref_mean+rewards_ref_std, alpha=0.5)
    ax[0].legend()

    width = 0.35
    stacks = 2
    x1 = [x - width/stacks, x + width/stacks]
    bottom = np.zeros(len(x))
    i = 0
    for item, values in data_dict.items():
        p = ax[1].bar(x1[0], values, width, label=item, bottom=bottom, color = colors_NN[i], hatch=patterns[i])
        bottom += values
        i += 1
    if ref_stats:
        bottom = np.zeros(len(x))
        i = 0
        for item, values in ref_data_dict.items():
            p = ax[1].bar(x1[1], values, width, label=item, bottom=bottom, color = colors_inf[i], hatch=patterns[i])
            bottom += values
            i += 1
    ax[1].legend(loc='upper left')
    ax[1].set_xlabel("number of steps")

    fig.suptitle(title, y=0.98)
    plt.draw()
    fig.savefig(filename)
    plt.close(fig)

    return

def plot_training_data_evolution(data, filename, title):
    """
    Plot performance vs training time
    """

    fig, ax = plt.subplots(1, 3, figsize=(20, 7.5))
    plt.subplots_adjust(
        left=0.05, bottom=0.24, right=0.99, top=0.9, wspace=0.15, hspace=0
    )
    palette = plt.get_cmap("tab10")
    k = 0
    kwargs0 = {"markersize": 5 / (k + 1), "linewidth": 2 / (k + 1), "color": palette(k)}

    for i in range(3):
        if i == 0:
            yvar = data[:, 1]
            yname = "eps"
        elif i == 1:
            yvar = data[:, 2]
            yname = "mean loss of mini-batch"
        elif i == 2:
            yvar = data[:, 3]
            yname = "mean total rewards of mini-batch"
        x = np.arange(len(yvar))
        (line,) = ax[i].plot(x, yvar, '-', **kwargs0)  # '-o' for adding dots 
        ax[i].set_title(yname)
        ax[i].set_xlabel("training iteration")
        N = int(len(yvar)/10)
        if i > 0:
            (line,) = ax[i].plot(np.arange(N/2, len(yvar)-N/2+1), np.convolve(yvar, np.ones(N)/N, mode='valid'), color='pink', label = f"{N} moving average")
    
    ax[1].legend()
    ax[2].legend()
    ax[0].set_ylim(bottom=0)

    fig.suptitle(title, y=0.98)
    plt.draw()
    fig.savefig(filename)
    plt.close(fig)

    return 

def save_stats(it, timestamp, stats, filename):
    """save stats to txt file"""

    pfile = open(filename, "a")
    for out in (None, pfile):
        print(f"time_{it} = " + str(timestamp), file=out)
        print(f"success_source_and_flux_rate_{it} = " + str(stats["success_source_and_flux_rate"]), file=out)
        print(f"success_only_source_rate_{it} = " + str(stats["success_only_source_rate"]), file=out)
        print(f"success_only_flux_rate_{it} = " + str(stats["success_only_flux_rate"]), file=out)
        print(f"converged_wrong_rate_{it} = " + str(stats["converged_wrong_rate"]), file=out)
        print(f"not_converged_rate_{it} = " + str(stats["not_converged_rate"]), file=out)
        print(f"total_rewards_mean_{it} = " + str(stats["total_rewards_mean"]), file=out)
        print(f"total_rewards_std_{it} = " + str(stats["total_rewards_std"]), file=out)
        print(f"total_rewards_median_{it} = " + str(stats["total_rewards_median"]), file=out)
        print(f"total_rewards_5percentile_{it} = " + str(stats["total_rewards_5percentile"]), file=out)
        print(f"total_rewards_25percentile_{it} = " + str(stats["total_rewards_25percentile"]), file=out)
        print(f"total_rewards_75percentile_{it} = " + str(stats["total_rewards_75percentile"]), file=out)
        print(f"total_rewards_95percentile_{it} = " + str(stats["total_rewards_95percentile"]), file=out)
        print(f"rewards_mean_{it} = " + str(stats["rewards_mean"]), file=out)
        print(f"rewards_std_{it} = " + str(stats["rewards_std"]), file=out)
        print(f"rewards_median_{it} = " + str(stats["rewards_median"]), file=out)
        print(f"rewards_5percentile_{it} = " + str(stats["rewards_5percentile"]), file=out)
        print(f"rewards_25percentile_{it} = " + str(stats["rewards_25percentile"]), file=out)
        print(f"rewards_75percentile_{it} = " + str(stats["rewards_75percentile"]), file=out)
        print(f"rewards_95percentile_{it} = " + str(stats["rewards_95percentile"]), file=out)
        print(f"DRPS_mean_flux_{it} = " + str(stats["DRPS_mean_flux"]), file=out)
        print(f"DRPS_mean_x_{it} = " + str(stats["DRPS_mean_x"]), file=out)
        print(f"DRPS_mean_y_{it} = " + str(stats["DRPS_mean_y"]), file=out)
        print(f"DRPS_std_flux_{it} = " + str(stats["DRPS_std_flux"]), file=out)
        print(f"DRPS_std_x_{it} = " + str(stats["DRPS_std_x"]), file=out)
        print(f"DRPS_std_y_{it} = " + str(stats["DRPS_std_y"]), file=out)
        print(f"DRPS_median_flux_{it} = " + str(stats["DRPS_median_flux"]), file=out)
        print(f"DRPS_median_x_{it} = " + str(stats["DRPS_median_x"]), file=out)
        print(f"DRPS_median_y_{it} = " + str(stats["DRPS_median_y"]), file=out)
        print(f"DRPS_5percentile_flux_{it} = " + str(stats["DRPS_5percentile_flux"]), file=out)
        print(f"DRPS_5percentile_x_{it} = " + str(stats["DRPS_5percentile_x"]), file=out)
        print(f"DRPS_5percentile_y_{it} = " + str(stats["DRPS_5percentile_y"]), file=out)
        print(f"DRPS_25percentile_flux_{it} = " + str(stats["DRPS_25percentile_flux"]), file=out)
        print(f"DRPS_25percentile_x_{it} = " + str(stats["DRPS_25percentile_x"]), file=out)
        print(f"DRPS_25percentile_y_{it} = " + str(stats["DRPS_25percentile_y"]), file=out)
        print(f"DRPS_75percentile_flux_{it} = " + str(stats["DRPS_75percentile_flux"]), file=out)
        print(f"DRPS_75percentile_x_{it} = " + str(stats["DRPS_75percentile_x"]), file=out)
        print(f"DRPS_75percentile_y_{it} = " + str(stats["DRPS_75percentile_y"]), file=out)
        print(f"DRPS_95percentile_flux_{it} = " + str(stats["DRPS_95percentile_flux"]), file=out)
        print(f"DRPS_95percentile_x_{it} = " + str(stats["DRPS_95percentile_x"]), file=out)
        print(f"DRPS_95percentile_y_{it} = " + str(stats["DRPS_95percentile_y"]), file=out)
        print(f"DRPS_rel_median_flux_{it} = " + str(stats["DRPS_rel_median_flux"]), file=out)
        print(f"DRPS_rel_median_x_{it} = " + str(stats["DRPS_rel_median_x"]), file=out)
        print(f"DRPS_rel_median_y_{it} = " + str(stats["DRPS_rel_median_y"]), file=out)
        print(f"DRPS_rel_75percentile_flux_{it} = " + str(stats["DRPS_rel_75percentile_flux"]), file=out)
        print(f"DRPS_rel_75percentile_x_{it} = " + str(stats["DRPS_rel_75percentile_x"]), file=out)
        print(f"DRPS_rel_75percentile_y_{it} = " + str(stats["DRPS_rel_75percentile_y"]), file=out)
        print(f"DRPS_rel_95percentile_flux_{it} = " + str(stats["DRPS_rel_95percentile_flux"]), file=out)
        print(f"DRPS_rel_95percentile_x_{it} = " + str(stats["DRPS_rel_95percentile_x"]), file=out)
        print(f"DRPS_rel_95percentile_y_{it} = " + str(stats["DRPS_rel_95percentile_y"]), file=out)

    return None