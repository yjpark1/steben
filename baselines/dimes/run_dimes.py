import json
import os
import pickle
import random
import shutil
import time
from dataclasses import dataclass
import networkx as nx
import copy

import numpy as np
import torch
import tyro
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

from baselines.dimes.agents import DimesAgent, DimesTorchAgent
from baselines.dimes.network import DimesNet

from baselines.cherrypick.networks import Vulcan

from baselines.dimes.env import STPEdgeEnvironment
from stpgen.solvers.heuristic import TwoApproximation
from baselines.dimes.util import new_env


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def evaluation(args, env, agent, global_step, writer, episodes=200, iterations=1, instance=None, graph=None):
    # load the best model
    path = os.path.join(args.log_dir, f"{args.filename_for_dqn_weights}.pth")
    agent.network.load_state_dict(torch.load(path))
    agent.network.eval()
    agent.epsilon = 0.
    
    total_reward_eval = []
    cost_eval = []
    gap_eval = []
    for episode in range(episodes):  # Run 100 episodes
        obs, infos = env.reset(instance=instance)
        solver = TwoApproximation(env.instance)
        solver.solve()
        solver.solution = env.instance.edge_subgraph(solver.solution.edges).copy()
        
        c_huer = np.sum(list(map(lambda x: x[2]['cost'], solver.solution.edges(data=True)))) / env.max_edge_cost
        inst = env.instance.copy()
        
        total_reward_eval_repeat = []
        cost_eval_repeat = []
        terminals = inst.graph['Terminals']['terminals']
        for ti in range(iterations):
            if bool(args.test_terminal):
                v_idx = ti % len(terminals)
            else:
                v_idx = None
            total_reward, cost = agent.eval_instance_with_heatmap(
                inst, initial_vertex_idx=v_idx, pruning=bool(args.pruning), select=args.test_method, temperature=args.temperature)

            total_reward_eval_repeat.append(total_reward)
            cost_eval_repeat.append(cost)

        idx = np.argmin(cost_eval_repeat)
        total_reward_eval.append(total_reward_eval_repeat[idx])
        cost_eval.append(cost_eval_repeat[idx])
        gap = cost_eval[-1] / c_huer
        gap_eval.append(gap)

    writer.add_scalar("eval/cost", np.mean(cost_eval), global_step)
    writer.add_scalar("eval/gap", np.mean(gap_eval), global_step)
    print(f"cost: {np.mean(cost_eval): .3f}, gap: {np.mean(gap_eval): .3f}")


def train(args, env, agent, writer):
    history = agent.train(writer)
    return history

def run(args, is_train=True):
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    print(log_dir)
    dict_path = os.path.join(log_dir, 'args.json')
    args_dict = {}
    for key, value in vars(args).items():
        if callable(value):
            continue
        elif key == 'device':
            continue
        args_dict[key] = value
    with open(dict_path, 'w') as f:
        f.write(json.dumps(args_dict, indent=2))

    # shutil.rmtree(f"runs/{args.run_name}", ignore_errors=True)
    writer = SummaryWriter(log_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env = new_env(args.n_nodes, args.seed, args, args.graph_type, args.cost_type, device)
    
    # agent setup
    agent = DimesAgent(DimesNet, args)
    # agent = DimesTorchAgent(DimesNet, args)
    # agent = DQNAgent(lambda: VulcanOrigin(), args)
    history = {}
    try:
        if is_train:
            history = train(args, env, agent, writer)
        else:
            path = f"{args.log_dir}/{args.filename_for_dqn_weights}.pth"
            agent.network.load_state_dict(torch.load(path))
            
        evaluation(args, env, agent, history['global_step'], writer, iterations=32)
    
    except KeyboardInterrupt:
        print("Interrupted by user, shutting down.")
        
    finally:
        if is_train:
            path = f"{log_dir}/learning_curve_dimes.pkl"
            with open(path, 'wb') as f:
                pickle.dump(history, f)

def args_prep(args):
    if getattr(args, 'n_nodes_min', None) is None:
        args.n_nodes_min = args.n_nodes
    if getattr(args, 'n_nodes_max', None) is None:
        args.n_nodes_max = args.n_nodes

    args.outer_opt_fn = lambda pars: getattr(optim, args.outer_opt)(pars, lr=args.outer_opt_lr, weight_decay=args.outer_opt_wd)
    args.inner_opt_fn = lambda pars: getattr(optim, args.inner_opt)(pars, lr=args.inner_opt_lr, weight_decay=args.inner_opt_wd)
    return args


if __name__ == "__main__":
    from pydev_check import check_pydevd
    check_pydevd()

    import sys
    sys.path.insert(0, './')
    sys.path.insert(0, '../../')

    import cProfile
    import pstats
    import datetime
    from baselines.dimes_tsp.inc.tsp_args import args_init
    profiler = cProfile.Profile()
    profiler.enable()

    is_train = True
    args = args_init()
    run(args, is_train)

    profiler.disable()
    profiler.dump_stats(f'{args.log_dir}profile.prof')

    # Load the profiling results and print the stats
    p = pstats.Stats('profile.prof')
    p.strip_dirs().sort_stats('cumulative').print_stats(10)
