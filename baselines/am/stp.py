from typing import Callable, Optional, Union

import networkx as nx
import numpy as np
import ray
import torch
from rl4co.envs.common.base import RL4COEnvBase
from rl4co.envs.common.utils import Generator, get_sampler
from rl4co.utils.ops import gather_by_index, get_tour_length
from rl4co.utils.pylogger import get_pylogger
from scipy.sparse.csgraph import dijkstra
from scipy.stats import binom
from tensordict.tensordict import TensorDict
from torch.distributions import Uniform
from torchrl.data import (BoundedTensorSpec, CompositeSpec,
                          UnboundedContinuousTensorSpec,
                          UnboundedDiscreteTensorSpec)

from baselines.common.utils import NodeFeature

log = get_pylogger(__name__)


def _convert_partial_solution_into_binary_tensor(partial_solution):
    num_rows, num_cols = partial_solution.size()        
    result = torch.zeros_like(partial_solution, dtype=torch.int)
    index_tensor = partial_solution
    # Mask to ignore -1 values (or other invalid values)
    valid_mask = index_tensor >= 0

    # Create row indices for broadcasting
    row_indices = torch.arange(num_rows).unsqueeze(1).expand_as(index_tensor).to(index_tensor.device)

    # Scatter to set the specified indices to 1
    result[row_indices[valid_mask], index_tensor[valid_mask]] = 1
    return result


def get_batched_neighbors_from_binary_vector(adj_matrices, nodes_binary_vectors):
    """
    Get the binary vectors of neighbor nodes for given nodes from the adjacency matrices in batches using PyTorch.

    Parameters:
    adj_matrices (torch.Tensor): The batched adjacency matrices of the graphs. Shape: (batch_size, num_nodes, num_nodes)
    nodes_binary_vectors (torch.Tensor): The batched binary vectors where 1 indicates the node is included. Shape: (batch_size, num_nodes)

    Returns:
    torch.Tensor: A 3D tensor where each sub-tensor corresponds to the binary vectors of neighbors for each node in a batch.
    """
    batch_size, num_nodes, _ = adj_matrices.shape
    # Create a mask for nodes in each batch
    mask = nodes_binary_vectors.unsqueeze(-1).bool()
    # Apply the mask to get neighbors
    neighbors = adj_matrices * mask
    neighbors = neighbors.sum(dim=1).bool()
    return neighbors


def get_ray_class(STPInstance):
    @ray.remote
    class STPInstance_rayactor(STPInstance):
        def __init__(self, n=20, max_edge_cost=2 ** 16, seed=None, **kwargs):
            super().__init__(n, max_edge_cost, seed, **kwargs)
            
        def sample(self, batch_size: int):
            samples = []
            for i in range(batch_size):
                instance = super().sample()
                
                adj_matrix = nx.adjacency_matrix(instance, weight='cost').todense()
                adj_tensor = torch.tensor(adj_matrix, dtype=torch.float)
                
                terminals_vector = torch.zeros(instance.number_of_nodes(), dtype=torch.int64)
                terminals_vector.scatter_(0, torch.tensor(instance.graph['Terminals']['terminals']), 1)
                if self.graph_model == 'Grid':
                    node_features = NodeFeature.cherrypick_agumented_coord(instance)
                else:
                    node_features = NodeFeature.cherrypick(instance, 
                                                            normalize_edge_cost=True, 
                                                            max_edge_cost=self.max_edge_cost)
                node_features = np.array([value for key, value in 
                                        sorted(node_features.items(), key=lambda x: x[0])])
                node_features = torch.tensor(node_features, dtype=torch.float)
                
                instance_tensor = {
                    'adj': adj_tensor, 
                    'terminals': terminals_vector,
                    'node_features': node_features
                    }
                samples.append(instance_tensor)
            
            adjs = torch.stack([sample['adj'] for sample in samples])
            terminals = torch.stack([sample['terminals'] for sample in samples])
            node_features = torch.stack([sample['node_features'] for sample in samples])
            
            return TensorDict(
                {
                    "adjs": adjs,
                    "terminals": terminals,
                    "node_features": node_features,
                },
                batch_size=batch_size,
            )

    return STPInstance_rayactor
        

@ray.remote
def merge_data(data_batches, nsamples):
    data = {
        'adjs': np.concatenate([x['adjs'] for x in data_batches], axis=0)[:nsamples],
        'terminals': np.concatenate([x['terminals'] for x in data_batches], axis=0)[:nsamples],
        'node_features': np.concatenate([x['node_features'] for x in data_batches], axis=0)[:nsamples],
    }    
    return data

class ParallelDataGenerator:
    def __init__(self, sampler_actor, n, max_edge_cost, num_actors=-1, batch_size=32, **kwargs):
        ray.init(ignore_reinit_error=True)
        self.sampler_actor = sampler_actor
        self.n = n
        self.max_edge_cost = max_edge_cost
        self.num_actors = num_actors
        self.batch_size = batch_size
        self.seed = kwargs.get('seed', None)
        self.rng = np.random.default_rng(self.seed).spawn(self.num_actors)
        self.actors = [sampler_actor.remote(n=self.n, 
                                            max_edge_cost=self.max_edge_cost, 
                                            rng=self.rng[i], **kwargs) for i in range(self.num_actors)]

    def sample(self, batchsize):
        num_batches = np.ceil(batchsize / self.batch_size).astype(int)
        # Generate data in parallel
        data_futures = []
        for i in range(num_batches):
            actor = self.actors[i % self.num_actors]
            data_futures.append(actor.sample.remote(self.batch_size))
        data_batches = ray.get(data_futures)

        # Merge the data
        merged_data = ray.get(merge_data.remote(data_batches, batchsize))        
        return merged_data
        

class STPGenerator(Generator):
    """Data generator for the Steiner Tree Problem (STP).
    """
    def __init__(
        self,
        instance_sampler: Callable = None,
        num_nodes: int = 20,
        max_edge_cost: int = 2**16,
        **kwargs
    ):
        self.num_nodes = num_nodes
        self.max_edge_cost = max_edge_cost
        self.kwargs = kwargs
        
        if instance_sampler is None:
            raise ValueError("instance_sampler must be provided")
        else:
            if kwargs.get('useray', False):
                print("instance_sampler is a remote function or actor.")
                instance_sampler = get_ray_class(instance_sampler)
                self.instance_sampler = ParallelDataGenerator(sampler_actor=instance_sampler, 
                                                              n=num_nodes, 
                                                              max_edge_cost=max_edge_cost, 
                                                              **kwargs)
            else:
                self.instance_sampler = instance_sampler(n=num_nodes, 
                                                        max_edge_cost=max_edge_cost,
                                                        **kwargs)
    
    def _generate(self, batch_size) -> TensorDict:
        assert len(batch_size) == 1, "batch_size must be a single value"
        
        data = self.instance_sampler.sample(batch_size[0])
        
        adjs = torch.from_numpy(data['adjs'] / self.max_edge_cost).float()
        terminals = torch.from_numpy(data['terminals']).int()
        node_features = torch.from_numpy(data['node_features']).float()        
        
        return TensorDict(
            {
                "adjs": adjs,
                "terminals": terminals,
                "node_features": node_features,
            },
            batch_size=batch_size,
        )
    

class STPEnv(RL4COEnvBase):
    """Traveling Salesman Problem (TSP) environment
    At each step, the agent chooses a city to visit. The reward is 0 unless the agent visits all the cities.
    In that case, the reward is (-)length of the path: maximizing the reward is equivalent to minimizing the path length.

    Observations:
        - locations of each customer.
        - the current location of the vehicle.

    Constrains:
        - the tour must return to the starting customer.
        - each customer must be visited exactly once.

    Finish condition:
        - the agent has visited all customers and returned to the starting customer.

    Reward:
        - (minus) the negative length of the path.

    Args:
        generator: TSPGenerator instance as the data generator
        generator_params: parameters for the generator
    """

    name = "stp"

    def __init__(
        self,
        generator: STPGenerator = None,
        generator_params: dict = {},
        **kwargs,
    ):
        super().__init__(**kwargs)
        if generator is None:
            generator = STPGenerator(**generator_params)
        self.generator = generator
        # self._make_spec(self.generator)

    @staticmethod
    def _step(td: TensorDict) -> TensorDict:
        current_node = td["action"]
        # update partial solution
        partial_solution = td["partial_solution"]
        for i in range(td["partial_solution"].size(0)):
            if not td['done'][i]:
                partial_solution[i][td["i"][i]] = current_node[i]
        
        partial_solution_binary = _convert_partial_solution_into_binary_tensor(td['partial_solution'])
        available = get_batched_neighbors_from_binary_vector(td['adjs'], partial_solution_binary)
        available = available.float() - partial_solution_binary
        available = available > 0
        
        terminal_remaining = (1 - partial_solution_binary) * td['terminals']
        contains_all_terminals = terminal_remaining.sum(dim=-1, keepdim=True) == 0
        
        # No more actions allowed if all terminals are visited
        idx_done = contains_all_terminals.squeeze(-1)
        available[idx_done, :] = False
        available[idx_done, current_node[idx_done]] = True
        
        reward = torch.zeros_like(contains_all_terminals, dtype=torch.float32)
        
        td.update(
            {
                "current_node": current_node,
                "i": td["i"] + 1,
                "action_mask": available,
                "reward": reward,
                "partial_solution": partial_solution,
                "done": contains_all_terminals,
            },
        )
        return td

    def _reset(self, td: Optional[TensorDict] = None, batch_size=None) -> TensorDict:
        # Initialize locations
        device = td.device
        adjs = td["adjs"]

        # We do not enforce loading from self for flexibility
        num_nodes = adjs.shape[-2]
        
        # Other variables
        current_node = torch.zeros((batch_size), dtype=torch.int64, device=device)
        
        available = torch.zeros(
            (*batch_size, num_nodes), dtype=torch.bool, device=device
        )  # 1 means not visited, i.e. action is allowed
        available[torch.where(td['terminals'] == 1)] = 1
        
        i = torch.zeros((*batch_size, 1), dtype=torch.int64, device=device)

        partial_solution = torch.ones((*batch_size, num_nodes), dtype=torch.int64, device=device) * -1
        
        return TensorDict(
            {
                "adjs": td['adjs'],
                "terminals": td['terminals'],
                "node_features": td['node_features'],
                "current_node": current_node,
                "i": i,
                "action_mask": available,
                "reward": torch.zeros((*batch_size, 1), dtype=torch.float32),
                "partial_solution": partial_solution,
                "done": torch.zeros((*batch_size, 1), dtype=torch.bool),
            },
            batch_size=batch_size,
        )

    def _make_spec(self, generator: STPGenerator):
        self.observation_spec = CompositeSpec(
            locs=BoundedTensorSpec(
                low=generator.min_loc,
                high=generator.max_loc,
                shape=(generator.num_loc, 2),
                dtype=torch.float32,
            ),
            first_node=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            current_node=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            i=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            action_mask=UnboundedDiscreteTensorSpec(
                shape=(generator.num_loc),
                dtype=torch.bool,
            ),
            shape=(),
        )
        self.action_spec = BoundedTensorSpec(
            shape=(1),
            dtype=torch.int64,
            low=0,
            high=generator.num_loc,
        )
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(1))
        self.done_spec = UnboundedDiscreteTensorSpec(shape=(1), dtype=torch.bool)

    def _get_reward(self, td, actions) -> TensorDict:
        adjacencies = torch.where(td['adjs'] == 0, float('inf'), td['adjs'])

        costs = torch.zeros(td['done'].size(0), dtype=torch.float32, device=td.device)
        for i in range(td["partial_solution"].size(0)):
            solution = td["partial_solution"][i]
            solution = solution[solution >= 0]
            adjacency = adjacencies[i]
            
            for j, current_action in enumerate(solution[1:]):
                solution_prev = solution[:(j+1)]
                costs[i] += adjacency[current_action, solution_prev].min()
        
        return - costs

    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor):
        """Check that solution is valid: nodes are visited exactly once"""
        pass

    @staticmethod
    def render(td: TensorDict, actions: torch.Tensor=None, ax = None):
        return render(td, actions, ax)

    

class STPEnv_old(RL4COEnvBase):
    """Traveling Salesman Problem (TSP) environment
    At each step, the agent chooses a city to visit. The reward is 0 unless the agent visits all the cities.
    In that case, the reward is (-)length of the path: maximizing the reward is equivalent to minimizing the path length.

    Observations:
        - locations of each customer.
        - the current location of the vehicle.

    Constrains:
        - the tour must return to the starting customer.
        - each customer must be visited exactly once.

    Finish condition:
        - the agent has visited all customers and returned to the starting customer.

    Reward:
        - (minus) the negative length of the path.

    Args:
        generator: TSPGenerator instance as the data generator
        generator_params: parameters for the generator
    """

    name = "stp"

    def __init__(
        self,
        generator: STPGenerator = None,
        generator_params: dict = {},
        **kwargs,
    ):
        super().__init__(**kwargs)
        if generator is None:
            generator = STPGenerator(**generator_params)
        self.generator = generator
        # self._make_spec(self.generator)

    @staticmethod
    def _step(td: TensorDict) -> TensorDict:
        current_node = td["action"]
        # update partial solution
        partial_solution = td["partial_solution"]
        for i in range(td["partial_solution"].size(0)):
            if not td['done'][i]:
                partial_solution[i][td["i"][i]] = current_node[i]
        # partial_solution_binary = _convert_partial_solution_into_binary_tensor(partial_solution)
        available = torch.zeros_like(td["action_mask"])

        # Efficiently find neighbors
        for i in range(available.size(0)):
            # Filter out -1 values and get the current nodes
            current_nodes = partial_solution[i][partial_solution[i] >= 0]
            # Find neighbors using the adjacency matrix
            if current_nodes.numel() > 0:
                neighbors = td["adjs"][i][current_nodes].sum(dim=0).bool()
                available[i] = neighbors
                available[i][current_nodes] = False
        
        partial_solution_binary = _convert_partial_solution_into_binary_tensor(td['partial_solution'])
        terminal_remaining = (1 - partial_solution_binary) * td['terminals']
        contains_all_terminals = terminal_remaining.sum(dim=-1, keepdim=True) == 0
        
        # No more actions allowed if all terminals are visited
        idx_done = contains_all_terminals.squeeze(-1)
        available[idx_done, :] = False
        available[idx_done, current_node[idx_done]] = True
        
        reward = torch.zeros_like(contains_all_terminals, dtype=torch.float32)
        
        td.update(
            {
                "current_node": current_node,
                "i": td["i"] + 1,
                "action_mask": available,
                "reward": reward,
                "partial_solution": partial_solution,
                "done": contains_all_terminals,
            },
        )
        return td

    def _reset(self, td: Optional[TensorDict] = None, batch_size=None) -> TensorDict:
        # Initialize locations
        device = td.device
        adjs = td["adjs"]

        # We do not enforce loading from self for flexibility
        num_nodes = adjs.shape[-2]
        
        # Other variables
        current_node = torch.zeros((batch_size), dtype=torch.int64, device=device)
        
        available = torch.zeros(
            (*batch_size, num_nodes), dtype=torch.bool, device=device
        )  # 1 means not visited, i.e. action is allowed
        available[torch.where(td['terminals'] == 1)] = 1
        
        i = torch.zeros((*batch_size, 1), dtype=torch.int64, device=device)

        partial_solution = torch.ones((*batch_size, num_nodes), dtype=torch.int64, device=device) * -1
        
        return TensorDict(
            {
                "adjs": td['adjs'],
                "terminals": td['terminals'],
                "node_features": td['node_features'],
                "current_node": current_node,
                "i": i,
                "action_mask": available,
                "reward": torch.zeros((*batch_size, 1), dtype=torch.float32),
                "partial_solution": partial_solution,
                "done": torch.zeros((*batch_size, 1), dtype=torch.bool),
            },
            batch_size=batch_size,
        )

    def _make_spec(self, generator: STPGenerator):
        self.observation_spec = CompositeSpec(
            locs=BoundedTensorSpec(
                low=generator.min_loc,
                high=generator.max_loc,
                shape=(generator.num_loc, 2),
                dtype=torch.float32,
            ),
            first_node=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            current_node=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            i=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            action_mask=UnboundedDiscreteTensorSpec(
                shape=(generator.num_loc),
                dtype=torch.bool,
            ),
            shape=(),
        )
        self.action_spec = BoundedTensorSpec(
            shape=(1),
            dtype=torch.int64,
            low=0,
            high=generator.num_loc,
        )
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(1))
        self.done_spec = UnboundedDiscreteTensorSpec(shape=(1), dtype=torch.bool)

    def _get_reward(self, td, actions) -> TensorDict:
        # Gather locations in order of tour and return distance between them (i.e., -reward)
        locs_ordered = gather_by_index(td["partial_solution"], actions)
        adjacencies = torch.where(td['adjs'] == 0, float('inf'), td['adjs'])

        costs = torch.zeros(td['done'].size(0), dtype=torch.float32, device=td.device)
        for i in range(td["partial_solution"].size(0)):
            solution = td["partial_solution"][i]
            solution = solution[solution >= 0]
            adjacency = adjacencies[i]
            
            for j, current_action in enumerate(solution[1:]):
                solution_prev = solution[:(j+1)]
                costs[i] += adjacency[current_action, solution_prev].min()
                
        return - costs

    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor):
        """Check that solution is valid: nodes are visited exactly once"""
        pass

    @staticmethod
    def render(td: TensorDict, actions: torch.Tensor=None, ax = None):
        return render(td, actions, ax)

