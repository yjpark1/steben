import glob
import pickle
from typing import Any, Optional, Union

import networkx as nx
import numpy as np
import pytorch_lightning as pl
import ray
import torch
import torch.nn.functional as F
from pytorch_lightning import loggers as pl_loggers
from rl4co.data.dataset import TensorDictDataset
from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.nn.env_embeddings.context import EnvContext
from rl4co.models.rl.reinforce.baselines import REINFORCEBaseline
from rl4co.models.zoo.am import AttentionModel, AttentionModelPolicy
from rl4co.models.zoo.am.policy import AttentionModelPolicy
from rl4co.utils.ops import gather_by_index
from rl4co.utils.trainer import RL4COTrainer
from tensordict import TensorDict
from torch import nn
from torch.utils.data import DataLoader
from baselines.common.utils import NodeFeature

from baselines.am.stp import (
    STPEnv, STPGenerator, _convert_partial_solution_into_binary_tensor)
from stpgen.datasets.synthetic.instance import (MAX_EDGE_COST,
                                                 STPInstance_erdos_renyi,
                                                 STPInstance_grid,
                                                 STPInstance_regular,
                                                 STPInstance_watts_strogatz)
from stpgen.solvers.heuristic import TwoApproximation, remove_inessentials
from lightning.pytorch.callbacks import ModelCheckpoint
import tqdm
import os
import time
import re


def sort_key(filename):
    match = re.search(r'batch_(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0


class ProcessorNetwork(nn.Module):
	def __init__(self, embedding_dim):
		super(ProcessorNetwork, self).__init__()
		self.fc = nn.Linear(2 * embedding_dim, embedding_dim)

	def forward(self, embeddings, cost_matrix):
		# Use adjacency matrix to filter neighbors and sum them
		degree = cost_matrix.bool().float().sum(dim=-1).unsqueeze(2)	# (batch_size, nodes, 1)
		neighbor_embeddings = embeddings.unsqueeze(2) * cost_matrix.unsqueeze(-1)
		sum_neighbors = neighbor_embeddings.sum(dim=1) / degree
		# Concatenate original embeddings with their neighborhood sums
		concatenated = torch.cat([embeddings, sum_neighbors], dim=-1)
		return F.relu(self.fc(concatenated))


class STPInitEmbedding(nn.Module):
    """Initial embedding for the STP.
    """
    def __init__(self, embed_dim, node_dim=2, linear_bias=True):
        super(STPInitEmbedding, self).__init__()
        # node_dim = 2  # cherrypick feature
        self.init_embed = nn.Linear(node_dim, embed_dim, linear_bias)
        self.neighbor_embed = ProcessorNetwork(embed_dim)

    def forward(self, td):
        out = self.init_embed(td["node_features"])
        out = self.neighbor_embed(out, td['adjs'])
        return out


class STPContext(EnvContext):
    """Context embedding for the Steiner Tree Problem (STP).
    Project the following to the embedding space:
        - average of terminal node embeddings
        - current node embedding
    """

    def __init__(self, embed_dim):
        super(STPContext, self).__init__(embed_dim, 2 * embed_dim)
        self.W_placeholder = nn.Parameter(
            torch.Tensor(self.embed_dim).uniform_(-1, 1)
        )
        
    def forward(self, embeddings, td):
        batch_size = embeddings.size(0)
        node_dim = (-1, )
                
        if td["i"][(0,) * td["i"].dim()].item() < 1:  # get first item fast
            context_embedding = self.W_placeholder[None, :].expand(
                batch_size, self.W_placeholder.size(-1)
            )
            
        else:
            index = _convert_partial_solution_into_binary_tensor(td['partial_solution'])
            context_embedding = self._averaged_gather_by_boolean_index(embeddings, index).view(batch_size, *node_dim)
        
        context_embedding = torch.cat(
            [
                context_embedding,
                self._get_average_embedding_for_remaining_terminals(embeddings, td),
            ],
            dim=-1,
        )
        return self.project_context(context_embedding)
    
    def _get_average_embedding_for_remaining_terminals(self, embeddings, td):
        # Assuming `summed_embeddings` is the Float tensor and `embeddings` is the Half tensor
        embeddings = embeddings.to(torch.float32)
        
        batch_size = embeddings.size(0)
        partial_solution_binary = _convert_partial_solution_into_binary_tensor(td['partial_solution'])
        terminal_remaining = (1 - partial_solution_binary) * td['terminals']
        terminal_indices = terminal_remaining.nonzero(as_tuple=True)
        
        # Initialize the result tensor to store summed embeddings
        summed_embeddings = torch.zeros((batch_size, embeddings.size(-1)), device=embeddings.device)
        # Scatter the embeddings based on the terminal indices
        summed_embeddings.index_add_(0, terminal_indices[0], embeddings[terminal_indices[0], terminal_indices[1]])
        # Compute the count of terminals for normalization
        counts = torch.zeros((batch_size, 1), device=embeddings.device).index_add_(0, terminal_indices[0], 
                                                                                   torch.ones_like(terminal_indices[0], 
                                                                                                   dtype=torch.float).unsqueeze(1))
        # Compute the average embeddings
        averaged_terminal_embeddings = summed_embeddings / counts
        averaged_terminal_embeddings[td['done'].view(-1)] = self.W_placeholder
        
        return averaged_terminal_embeddings
    
    @staticmethod
    def _averaged_gather_by_boolean_index(embeddings, index):
        # Expand dimensions of index tensor to match embeddings tensor
        index_expanded = index.unsqueeze(-1).expand_as(embeddings)

        # Multiply embeddings with index tensor
        selected_embeddings = embeddings * index_expanded.float()

        # Sum along the node dimension
        sum_embeddings = selected_embeddings.sum(dim=1)

        # Count non-zero indices in each batch
        count = index.sum(dim=1, keepdim=True).clamp(min=1)  # Avoid division by zero

        # Average
        # average_embeddings = sum_embeddings / count.float()
        average_embeddings = sum_embeddings
        return average_embeddings


class StaticEmbedding(nn.Module):
    """Static embedding for general problems.
    This is used for problems that do not have any dynamic information, except for the
    information regarding the current action (e.g. the current node in TSP). See context embedding for more details.
    """

    def __init__(self, *args, **kwargs):
        super(StaticEmbedding, self).__init__()

    def forward(self, td):
        return 0, 0, 0


class AttentionModelSTP(AttentionModel):
    """Attention Model for the Steiner Tree Problem (STP).
    Check :class:`AttentionModel` for more details such as additional parameters including batch size.

    Args:
        env: Environment to use for the algorithm
        policy: Policy to use for the algorithm
        baseline: REINFORCE baseline. Defaults to rollout (1 epoch of exponential, then greedy rollout baseline)
        policy_kwargs: Keyword arguments for policy
        baseline_kwargs: Keyword arguments for baseline
        **kwargs: Keyword arguments passed to the superclass
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: AttentionModelPolicy = None,
        baseline: Union[REINFORCEBaseline, str] = "rollout",
        policy_kwargs={},
        baseline_kwargs={},
        **kwargs,
    ):
        if policy is None:
            policy = AttentionModelPolicy(env_name=env.name, **policy_kwargs)

        super().__init__(env, policy, baseline, baseline_kwargs, **kwargs)
        
    def shared_step(
        self, batch: Any, batch_idx: int, phase: str, dataloader_idx: int = None
    ):
        td = self.env.reset(batch)
        # Perform forward pass (i.e., constructing solution and computing log-likelihoods)
        out = self.policy(td, self.env, phase=phase)
        
        # Compute loss
        if phase == "train":
            out = self.calculate_loss(td, batch, out)
        metrics = self.log_metrics(out, phase, dataloader_idx=dataloader_idx)
        return {"loss": out.get("loss", None), **metrics}
        
    def calculate_loss(
        self,
        td: TensorDict,
        batch: TensorDict,
        policy_out: dict,
        reward: Optional[torch.Tensor] = None,
        log_likelihood: Optional[torch.Tensor] = None,
    ):
        """Calculate loss for REINFORCE algorithm.

        Args:
            td: TensorDict containing the current state of the environment
            batch: Batch of data. This is used to get the extra loss terms, e.g., REINFORCE baseline
            policy_out: Output of the policy network
            reward: Reward tensor. If None, it is taken from `policy_out`
            log_likelihood: Log-likelihood tensor. If None, it is taken from `policy_out`
        """
        # Extra: this is used for additional loss terms, e.g., REINFORCE baseline
        extra = batch.get("extra", None)
        reward = reward if reward is not None else policy_out["reward"]
        log_likelihood = (
            log_likelihood if log_likelihood is not None else policy_out["log_likelihood"]
        )
        # seqlen = (td['partial_solution'] > -1).sum(-1)
        # log_likelihood /= seqlen.float()
        
        # REINFORCE baseline
        bl_val, bl_loss = (
            self.baseline.eval(td, reward, self.env) if extra is None else (extra, 0)
        )
        
        # Main loss function
        advantage = reward - bl_val  # advantage = reward - baseline
        advantage /= td['terminals'].sum(-1) 
        advantage /= advantage.mean()
        
        reinforce_loss = -(advantage * log_likelihood).mean()
        loss = reinforce_loss + bl_loss
        policy_out.update(
            {
                "loss": loss,
                "reinforce_loss": reinforce_loss,
                "bl_loss": bl_loss,
                "bl_val": bl_val.mean(),
                "advantage_avg": advantage.mean(),
                "likelihood_avg": log_likelihood.mean(),
            }
        )
        
        return policy_out


def main(args):
    dim_nodefeature = 2
    # RL4CO env based on TorchRL
    if args.graph_type == "ER":
        instance_sampler = STPInstance_erdos_renyi
        max_edge_cost=2**16
        
    elif args.graph_type == "Grid":
        instance_sampler = STPInstance_grid
        dim_nodefeature = 4
        max_edge_cost=1.
        
    elif args.graph_type == "RR":
        instance_sampler = STPInstance_regular
        max_edge_cost=2**16
        
    elif args.graph_type == "WS":
        instance_sampler = STPInstance_watts_strogatz
        max_edge_cost=2**16
        
    else:
        raise ValueError(f"Unknown graph type: {args.graph_type}")
    
    generator=STPGenerator(instance_sampler=instance_sampler, 
                           num_nodes=args.num_nodes, max_edge_cost=max_edge_cost, seed=args.seed, useray=True,
                           cost_type=args.edge_cost_type,
                           num_actors=args.num_cores)
    env = STPEnv(generator)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    td_init = env.reset(batch_size=[3]).to(device)
    
    # Policy: neural network, in this case with encoder-decoder architecture
    policy = AttentionModelPolicy(env_name=env.name, 
                                embed_dim=128,
                                num_encoder_layers=3,
                                num_heads=8,
                                init_embedding=STPInitEmbedding(128, dim_nodefeature),
                                context_embedding=STPContext(128),
                                dynamic_embedding=StaticEmbedding(),
                                )
    if args.is_train:
        # Model: default is AM with REINFORCE and greedy rollout baseline
        model = AttentionModelSTP(env, 
                            policy=policy,
                            baseline=args.baseline,
                            batch_size=args.batch_size,
                            train_data_size=args.train_data_size,
                            val_data_size=args.batch_size * 2,
                            test_data_size=1,
                            optimizer_kwargs={"lr": 1e-4},
                            dataloader_num_workers=0,
                            metrics={'train': ['loss', 'reward', 'reinforce_loss', 'bl_val',
                                               'advantage_avg', 'likelihood_avg'],},
                            ) 

        # Greedy rollouts over untrained policy
        policy = policy.to(device)
        out = policy(td_init.clone(), env=env, phase="test", decode_type="greedy", return_actions=True)
        actions_untrained = out['actions'].cpu().detach()
        rewards_untrained = out['reward'].cpu().detach()

        for i in range(3):
            print(f"Problem {i+1} | Cost: {-rewards_untrained[i]:.3f}")
            # env.render(td_init[i], actions_untrained[i])
        
        log_dir = f'logs/{args.graph_type}@{args.num_nodes}@{args.edge_cost_type}/version_{args.version}'
        if os.path.exists(log_dir) and args.resume_from_checkpoint is None:
            for file in glob.glob(f'{log_dir}/events.out*'):
                os.remove(file)
        tb_logger = pl_loggers.TensorBoardLogger(f'logs/', 
                                                 name=f'{args.graph_type}@{args.num_nodes}@{args.edge_cost_type}',
                                                 version=args.version)
        
        # Define a checkpoint callback
        checkpoint_callback = [
            ModelCheckpoint(
                monitor='val/reward',  # Monitor validation loss
                dirpath=f'{log_dir}/bestcheckpoint',  # Directory where checkpoints will be saved
                save_top_k=1,  # Save only the best model
                mode='max',  # 'min' for loss and 'max' for metrics
                verbose=True)
        ] if args.is_train else None
            
        trainer = RL4COTrainer(
            max_epochs=args.num_epochs,
            accelerator="gpu",
            devices=[args.gpu_id] if type(args.gpu_id) is int else args.gpu_id,
            logger=tb_logger,
            log_every_n_steps=50,
            callbacks=checkpoint_callback,
            gradient_clip_val=1.0,
            reload_dataloaders_every_n_epochs=10,
        )
        
        trainer.fit(model, ckpt_path=args.resume_from_checkpoint)



class EvaluateModel:
    def __init__(self, args):
        ray.init(ignore_reinit_error=True)
        self.args = args
        
    @torch.no_grad()
    def evaluate(self):
        batch_size = 2000 
        if self.args.use_nsamples is not None:
            batch_size = self.args.use_nsamples
            
        if self.args.num_nodes > 200:
            batch_size = 10
        elif self.args.num_nodes > 50:
            batch_size = 100

        dim_nodefeature = 2
        # RL4CO env based on TorchRL
        if self.args.graph_type == "ER":
            instance_sampler = STPInstance_erdos_renyi
            max_edge_cost=2**16
            
        elif self.args.graph_type == "Grid":
            instance_sampler = STPInstance_grid
            dim_nodefeature = 4
            max_edge_cost=1.
            
        elif self.args.graph_type == "RR":
            instance_sampler = STPInstance_regular
            max_edge_cost=2**16
            
        elif self.args.graph_type == "WS":
            instance_sampler = STPInstance_watts_strogatz
            max_edge_cost=2**16
            
        else:
            raise ValueError(f"Unknown graph type: {self.args.graph_type}")
        
        generator=STPGenerator(instance_sampler=instance_sampler, 
                                num_nodes=self.args.num_nodes, max_edge_cost=max_edge_cost, seed=self.args.seed, useray=True,
                                cost_type=self.args.edge_cost_type,
                                num_actors=self.args.num_cores)
        env = STPEnv(generator)
        device = device = torch.device(f"cuda:{self.args.gpu_id}" if torch.cuda.is_available() else "cpu")
        
        model = AttentionModelSTP.load_from_checkpoint(env=env, checkpoint_path=self.args.checkpoint,
                                                        train_data_size=32,
                                                        val_data_size=32,
                                                        test_data_size=32,
                                                        map_location=device,
                                                        load_baseline=False,
                                                        )
        policy = model.policy
        policy.eval()
        policy.to(device)
        
        if self.args.inference_iterations > 1:
            costs, gaps, duration = self.eval_multiple(policy, env, device, batch_size)
        else:
            costs, gaps, duration = self.eval_single(policy, env, device, batch_size)
            
        print('gap:', np.mean(gaps), 'std:', np.std(gaps), 'samples:', len(gaps), 'duration:', duration)
        filepath = self.args.checkpoint.replace('.ckpt', f'@inference@{self.args.inference_decode_type}@{self.args.path_inference_save}.pkl')
        with open(filepath, "wb") as f:
            pickle.dump(costs, f)
    
    def eval_single(self, policy, env, device, batch_size):
        results = []
        gaps = []
        duration = 0
        num_cumulative = 0
        with tqdm.tqdm(total=self.args.use_nsamples) as pbar:
            for i, path_data in enumerate(sorted(glob.glob(self.args.testdata_dir + "/*.pkl"), key=sort_key)):
                with open(path_data, "rb") as f:
                    data = pickle.load(f)[:(self.args.use_nsamples - num_cumulative)]
                    td = self.make_td_from_graphs(data)
                    dl = DataLoader(dataset=TensorDictDataset(td), batch_size=batch_size, shuffle=False, num_workers=0)
                    
                    t0 = time.time()
                    for batch_idx, batch in enumerate(dl):
                        nsamples = batch['terminals'].size(0)
                        batch = TensorDict(batch, batch_size=nsamples, device=device)
                        td_init = env.reset(batch).clone().to(device)
                        out = policy(td_init, env=env, phase="test", decode_type=self.args.inference_decode_type, return_actions=True)
                        actions = out['actions'].cpu().numpy()
                        
                        for j, action in enumerate(actions):
                            idx = j + (batch_size * batch_idx)
                            _, problem, solution = data[idx]
                            subgraph, c = self.get_subgraph_from_state(problem, action)
                            cost_opt = solution.graph['Info']['cost']
                            results.append(c)
                            gaps.append(c / cost_opt)
                        num_cumulative += nsamples
                        pbar.update(nsamples)
                        if num_cumulative >= self.args.use_nsamples:
                            break
                duration += (time.time() - t0)
                if num_cumulative >= self.args.use_nsamples:
                    break
            
        return results, gaps, duration 
        
    def eval_multiple(self, policy, env, device, batch_size=10000):
        results = []
        gaps = []
        duration = 0
        for iteration in tqdm.tqdm(range(self.args.inference_iterations)):
            r, g, d = self.eval_single(policy, env, device, batch_size=batch_size)
            duration += d
            results.append(r)
            gaps.append(g)
        
        results = np.array(results)
        gaps = np.array(gaps)
        
        idx = np.array(results).argmin(axis=0)
        return results[idx], gaps[idx], duration
        
    def make_td_from_graphs(self, data):
        futures = []
        for ind, instance, solution in data:
            if 'steinlib' in self.args.testdata_dir.lower():
                futures.append(make_td_from_instance_for_steinlib.remote(instance))
            else:
                futures.append(make_td_from_instance.remote(instance, is_grid=self.args.graph_type == 'Grid'))
                
            ########
            # futures.append(make_td_from_instance.remote(instance))
            
            # adj_matrix = nx.adjacency_matrix(instance, weight='cost').todense()
            # adj_tensor = torch.tensor(adj_matrix, dtype=torch.float)
            # maxval = adj_tensor.max().item()
            
            # terminals_vector = torch.zeros(instance.number_of_nodes(), dtype=torch.int64)
            # terminals_vector.scatter_(0, torch.tensor(instance.graph['Terminals']['terminals']), 1)
            
            # node_features = NodeFeature.cherrypick(instance, 
            #                                         normalize_edge_cost=True, max_edge_cost=maxval)
            # node_features = np.array([value for key, value in 
            #                         sorted(node_features.items(), key=lambda x: x[0])])
            # node_features = torch.tensor(node_features, dtype=torch.float)
            
            # instance_tensor = {
            #     'adj': adj_tensor, 
            #     'terminals': terminals_vector,
            #     'node_features': node_features
            #     }
            # samples.append(instance_tensor)
            #########
        samples = []
        for td in ray.get(futures):
            samples.append(td)
        
        adjs = torch.stack([sample['adj'] for sample in samples])
        terminals = torch.stack([sample['terminals'] for sample in samples])
        node_features = torch.stack([sample['node_features'] for sample in samples])
            
        
        batch_size = len(data)
            
        td = TensorDict(
                {
                    "adjs": adjs,
                    "terminals": terminals,
                    "node_features": node_features,
                },
                batch_size=batch_size,
            )
        return td

    @staticmethod
    def extract_cost_from_data(data):
        costs = []
        for ind, instance, solution in data:
            cost = solution.graph['Info']['cost'] / solution.graph['Info']['max_edge_cost']
            costs.append(cost)
        return costs
    
    def get_subgraph_from_state(self, problem, action):
        graph = problem
        solution = self.remove_duplicates_preserve_order(action)
        edges_sol = []
        if len(solution) > 1:
            for i, current_action in enumerate(solution[1:]):
                partial_solution_prev = solution[:(i+1)]
                edges = [((current_action, n), graph[current_action][n]) for n in partial_solution_prev if graph.has_edge(current_action, n)]
                edge_selected = sorted(edges, key=lambda x: x[1]['cost'])[0]
                edges_sol.append(edge_selected[0])
        
        subgraph = graph.edge_subgraph(edges_sol)
        terminals = graph.graph['Terminals']['terminals']
        subgraph = remove_inessentials(subgraph.copy(), terminals)
        
        cost = 0 
        for _, _, dt in subgraph.edges(data=True):
            cost += dt['cost']
        
        return subgraph, cost
    
    @staticmethod
    def remove_duplicates_preserve_order(lst):
        seen = set()
        result = []
        for item in lst:
            if item not in seen:
                result.append(item)
                seen.add(item)
        return result
    
    @torch.no_grad()
    def eval_steinlib(self):
        instance_sampler = STPInstance_erdos_renyi
        generator = STPGenerator(instance_sampler=instance_sampler, 
                            num_nodes=self.args.num_nodes, max_edge_cost=2**16, seed=self.args.seed, useray=True, num_actors=1)
        env = STPEnv(generator)
        device = device = torch.device(f"cuda:{self.args.gpu_id}" if torch.cuda.is_available() else "cpu")
        
        model = AttentionModelSTP.load_from_checkpoint(env=env, checkpoint_path=self.args.checkpoint,
                                                        train_data_size=32,
                                                        val_data_size=32,
                                                        test_data_size=32,
                                                        map_location=device,
                                                        load_baseline=False,
                                                        )
        policy = model.policy
        policy.eval()
        policy.to(device)
        
        
        costs, gaps, duration = self.eval_single_steinlib(policy, env, device, batch_size=1)
        print('gap:', np.mean(gaps), 'std:', np.std(gaps), 'samples:', len(gaps), 'duration:', duration)
        filepath = self.args.checkpoint.replace('.ckpt', f'@inference@{self.args.inference_decode_type}@{self.args.path_inference_save}.pkl')
        with open(filepath, "wb") as f:
            pickle.dump(costs, f)
            
    def eval_single_steinlib(self, policy, env, device, batch_size=1):
        results = []
        gaps = []
        duration = 0
        num_cumulative = 0
        t0 = time.time()
        with tqdm.tqdm(total=10) as pbar:
            path_data = self.args.testdata_dir
            with open(path_data, "rb") as f:
                data = pickle.load(f)
                for sample in data:
                    td = self.make_td_from_graphs([sample])
                    batch = TensorDict(td, batch_size=1, device=device)
                    td_init = env.reset(batch).clone().to(device)
                    out = policy(td_init, env=env, phase="test", decode_type=self.args.inference_decode_type, return_actions=True)
                    action = out['actions'].cpu().numpy()[0]
                    
                    _, problem, solution = sample
                    subgraph, c = self.get_subgraph_from_state(problem, action)
                    cost_opt = problem.graph['Info']['cost']
                    results.append(c)
                    gaps.append(c / cost_opt)
                    pbar.update(1)
                duration += (time.time() - t0)
            
        return results, gaps, duration 

@ray.remote
def make_td_from_instance(instance, is_grid=False):
    adj_matrix = nx.adjacency_matrix(instance, weight='cost').todense()
    adj_tensor = torch.tensor(adj_matrix, dtype=torch.float)
    
    terminals_vector = torch.zeros(instance.number_of_nodes(), dtype=torch.int64)
    terminals_vector.scatter_(0, torch.tensor(instance.graph['Terminals']['terminals']), 1)
    
    if is_grid:
        maxcost = 1.
        node_features = NodeFeature.cherrypick_agumented_coord(instance)
    else:
        maxcost = 2 ** 16
        node_features = NodeFeature.cherrypick(instance, 
                                                normalize_edge_cost=True)
    
    node_features = np.array([value for key, value in 
                            sorted(node_features.items(), key=lambda x: x[0])])
    node_features = torch.tensor(node_features, dtype=torch.float)
    
    instance_tensor = {
        'adj': adj_tensor / maxcost, 
        'terminals': terminals_vector,
        'node_features': node_features
        }
    return instance_tensor


@ray.remote
def make_td_from_instance_for_steinlib(instance):
    adj_matrix = nx.adjacency_matrix(instance, weight='cost').todense()
    adj_tensor = torch.tensor(adj_matrix, dtype=torch.float)
    maxval = adj_tensor.max().item()
    
    terminals_vector = torch.zeros(instance.number_of_nodes(), dtype=torch.int64)
    terminals_vector.scatter_(0, torch.tensor(instance.graph['Terminals']['terminals']), 1)
    
    node_features = NodeFeature.cherrypick(instance, 
                                            normalize_edge_cost=True, 
                                            max_edge_cost=maxval)
    node_features = np.array([value for key, value in 
                            sorted(node_features.items(), key=lambda x: x[0])])
    node_features = torch.tensor(node_features, dtype=torch.float)
    
    instance_tensor = {
        'adj': adj_tensor / maxval, 
        'terminals': terminals_vector,
        'node_features': node_features
        }
    return instance_tensor