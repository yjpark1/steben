from dataclasses import dataclass
import torch


@dataclass
class Args:
    seed: int = 1234
    """The seed used for random number generation in the experiment."""

    torch_deterministic: bool = True
    """Determines whether PyTorch operations should be deterministic. If True, `torch.backends.cudnn.deterministic` is set to False."""

    cuda: bool = False
    """Determines whether CUDA should be enabled by default."""

    learning_rate: float = 1e-4
    """The learning rate used by the optimizer."""

    num_envs: int = 1
    """The number of parallel game environments."""

    buffer_size: int = 100000
    """The size of the replay memory buffer."""

    gamma: float = 0.99
    """The discount factor gamma used in the Q-learning algorithm."""

    tau: float = 1.0
    """The rate at which the target network is updated."""

    target_network_frequency: int = 500
    """The number of timesteps between updates of the target network."""

    batch_size: int = 128
    """The batch size used when sampling from the replay memory."""

    start_e: float = 0.1
    """The initial epsilon value used for the epsilon-greedy exploration strategy."""

    end_e: float = 0
    """The final epsilon value used for the epsilon-greedy exploration strategy."""

    exploration_fraction: float = 0.8
    """The fraction of total timesteps over which the epsilon value decreases from start_e to end_e."""

    learning_starts: int = 1000
    """The timestep at which learning begins."""

    train_frequency: int = 4
    """The frequency at which training occurs."""

    graph_type: str = 'ER'
    """The type of graph used in the experiment."""

    edge_cost_type: str = 'gaussian'
    """The type of edge cost used in the experiment."""

    num_nodes: int = 20
    """The number of nodes in the graph."""

    reward_function: str = 'default'
    """The reward function used in the experiment."""

    testdata_dir: str = None
    """The directory where test data is stored."""

    num_inferences: int = 1
    """The number of inferences made per instance."""

    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
    """The device on which computations are performed."""

    run_name: str = 'test'
    """The name of the run."""

    num_max_episodes: int = 6000
    """The maximum number of training instances."""

    num_patience: int = 250
    """The number of inferences made per instance."""

    filename_for_dqn_weights: str = "dqn_weight"
    """The filename used to save the DQN weights."""

    log_dir: str = f"logs/runs/dqn"
    """The directory where logs are saved."""

    path_inference_save: str = ''
    """The path where inference results are saved."""


if __name__ == "__main__":
    import sys
    sys.path.insert(0, './')
    
    from baselines.cherrypick.run import run
    
    # training
    args = Args(log_dir="baselines/cherrypick/checkpoints", graph_type="ER", num_nodes=10,
                filename_for_dqn_weights="dqn_weight@ER10g")
    run(args, is_train=True)
    
    # testing
    args = Args(log_dir="baselines/cherrypick/checkpoints", graph_type="ER", num_nodes=10, 
                filename_for_dqn_weights="dqn_weight@ER10g")
    run(args, is_train=False)