import os
import time
import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import pickle

from baselines.cherrypick.train import train
from baselines.cherrypick.evaluate import evaluation
from baselines.cherrypick.evaluate_par import inference_and_evaluation
                                               
from baselines.cherrypick.env import STPEnvironment
from stpgen.datasets.synthetic.instance import (MAX_EDGE_COST,
                                                 STPInstance_erdos_renyi,
                                                 STPInstance_grid,
                                                 STPInstance_regular,
                                                 STPInstance_watts_strogatz)
from baselines.cherrypick.agents import DQNAgent
from baselines.cherrypick.networks import Vulcan


def run(args, is_train=True):
    if is_train:
        # args.log_dir = f"{args.log_dir}@{time.strftime('%m%d_%H%M%S')}"
        log_dir = args.log_dir
        os.makedirs(log_dir, exist_ok=True)
        print(log_dir)
        
        writer = SummaryWriter(log_dir)
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
    else:
        log_dir = args.log_dir
    
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # env setup
    if args.graph_type == "ER":
        instance_sampler = STPInstance_erdos_renyi(n=args.num_nodes, seed=args.seed, cost_type=args.edge_cost_type)
    elif args.graph_type == "Grid":
        instance_sampler = STPInstance_grid(n=args.num_nodes, seed=args.seed)
    elif args.graph_type == "RR":
        instance_sampler = STPInstance_regular(n=args.num_nodes, seed=args.seed, cost_type=args.edge_cost_type)
    elif args.graph_type == "WS":
        instance_sampler = STPInstance_watts_strogatz(n=args.num_nodes, seed=args.seed, cost_type=args.edge_cost_type)
        
    env = STPEnvironment(instance_sampler, args=args)
    
    # agent setup
    if args.graph_type == "Grid":
        feature_dim = 4
    else:
        feature_dim = 2
        
    agent = DQNAgent(lambda: Vulcan(p=64, k=feature_dim), args)
    if args.cuda:
        agent.network.to(agent.device)
    
    try:
        if is_train:
            history = train(args, env, agent, writer)
        else:
            path = f"{args.log_dir}/{args.filename_for_dqn_weights}.pth"
            agent.network.load_state_dict(torch.load(path))
            
        evaluation(args, env, agent, iterations=1)
        inference_and_evaluation(args, env, agent, iterations=args.num_inferences)
    
    except KeyboardInterrupt:
        print("Interrupted by user, shutting down.")
        
    finally:
        if is_train:
            path = f"{log_dir}/learning_curve_{args.filename_for_dqn_weights}.pkl"
            with open(path, 'wb') as f:
                pickle.dump(history)