from dataclasses import dataclass
from baselines.am.am import main as am_stp_main, EvaluateModel
import glob


@dataclass
class Args:
    seed: int = 1234
    """The seed used for random number generation in the experiment."""

    # User defined
    graph_type: str = 'ER'
    """The type of graph used in the experiment. Options are 'ER', 'RR', 'WS', 'Grid'."""

    edge_cost_type: str = 'gaussian'
    """The type of edge cost used in the experiment. Options are 'uniform', 'gaussian', 'unifrom'."""

    num_nodes: int = 10
    """The number of nodes in the graph."""

    num_cores: int = 12
    """The number of cores used for parallel processing."""

    is_train: bool = True
    """Determines whether the model should be trained."""

    num_epochs: int = 100
    """The number of training epochs."""

    gpu_id: int = 0
    """The ID of the GPU used for computations."""

    batch_size: int = 512
    """The batch size used when sampling from the replay memory."""

    train_data_size: int = 1_280_000
    """The size of the training data."""

    baseline: str = 'rollout'
    """The baseline used for comparison."""

    # Inference options
    inference_decode_type: str = 'greedy'
    """The type of decoding used during inference. Options are 'greedy', 'beam'."""
    
    inference_iterations: int = 1
    """The number of iterations used during inference."""

    # Inference args
    version: int = 0
    """The version of the model."""

    checkpoint: str = glob.glob(f"logs/{graph_type}@{num_nodes}@{edge_cost_type}/version_{version}/checkpoints/*.ckpt")[0]
    """The path to the checkpoint file."""

    testdata_dir: str = None
    """The directory where test data is stored."""
    
    path_inference_save: str = ""
    """The path where inference results are saved."""

    use_nsamples: int = 10000
    """The number of samples used for test dataset."""


if __name__ == "__main__":
    # training
    args = Args(gpu_id=0, graph_type='ER', num_nodes=10, testdata_dir="testdata_gaussian/synthetic_STP_dataset_test_erdos_renyi_n10_scipjack/")
    am_stp_main(args)
    
    # testing
    args.is_train = False
    args.checkpoint = 'baselines/am/checkpoints/ER10g.ckpt'
    EvaluateModel(args).evaluate()
    
    