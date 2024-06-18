# CherryPick Baseline

This directory contains the implementation of the CherryPick algorithm.

## Usage

To run the CherryPick algorithm, use the `run_cherrypick.py` script. Here is a basic example:

``` python
from dataclasses import dataclass
import torch
from baselines.cherrypick.run import run

@dataclass
class Args:
    seed: int = 1234
    torch_deterministic: bool = True
    cuda: bool = False
    learning_rate: float = 1e-4
    num_envs: int = 1
    buffer_size: int = 100000
    gamma: float = 0.99
    tau: float = 1.0
    target_network_frequency: int = 500
    batch_size: int = 128
    start_e: float = 0.1
    end_e: float = 0
    exploration_fraction: float = 0.8
    learning_starts: int = 1000
    train_frequency: int = 4
    graph_type: str = 'ER'
    edge_cost_type: str = 'gaussian'
    num_nodes: int = 20
    reward_function: str = 'default'
    testdata_dir: str = None
    num_inferences: int = 1
    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
    run_name: str = 'test'
    num_max_episodes: int = 6000
    num_patience: int = 250
    filename_for_dqn_weights: str = "dqn_weight"
    log_dir: str = f"logs/runs/dqn"
    path_inference_save: str = ''


# Training
args = Args(log_dir="baselines/cherrypick/checkpoints", graph_type="ER", num_nodes=10,
            filename_for_dqn_weights="dqn_weight@ER10")
run(args, is_train=True)

# Testing
args = Args(log_dir="baselines/cherrypick/checkpoints", graph_type="ER", num_nodes=10, 
            filename_for_dqn_weights="dqn_weight@ER10")
run(args, is_train=False)
```