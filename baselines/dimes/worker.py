import datetime

import ray
from ray.rllib.utils.compression import pack, unpack, unpack_if_needed

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from stpgen.datasets.synthetic.instance import STPInstance_erdos_renyi
from stpgen.solvers.heuristic import TwoApproximation, remove_inessentials
from baselines.cherrypick.env import STPEnvironment
from baselines.dimes.env import STPEdgeEnvironment
from baselines.dimes_tsp.inc.utils import torch_add_grad
from baselines.cherrypick.networks import GraphEmbedding
from baselines.dimes.util import *
import pydevd_pycharm


class Worker:
    def __init__(self, network_cls, args, seed, pydev_port=None):
        if pydev_port is not None:
            try:
                pydevd_pycharm.settrace('localhost', port=int(pydev_port), stdoutToServer=True, stderrToServer=True, suspend=False)
                print(f'Debugger: Connected to {pydev_port}')
            except:
                pass

        self.args = args
        self.device = torch.device("cpu")
        args.device = self.device
        self.neg_inf = torch.full((1,), -np.inf, dtype=torch.float32, device=self.device)
        self.network = network_cls(args)
        self.env = new_env(args.n_nodes, seed, args, args.graph_type, args.cost_type, self.device)
        self.tr_gamma = args.tr_gamma

        self.cumulative_reward = None
        self.rewards = None
        self.logits = None
        self.available_edges = None
        self.actions = None
        self.costs = None

    def load_state_dict(self, state_dict):
        state_dict = unpack_if_needed(state_dict)
        self.network.load_state_dict(state_dict)
        self.network.eval()

    @torch.no_grad()
    def rollout(self, heat_map, instance=None, select='gumbel', initial_vertex_idx=None, temperature=1.0, pruning=False):
        state, info = self.env.reset(instance=unpack_if_needed(instance), initial_vertex_idx=initial_vertex_idx)
        terminated = False
        par_adv = (heat_map - heat_map.mean())

        cumulative_reward = 0
        rewards = []
        logits = []
        probs = []
        available_edges = []
        actions = []

        while not terminated:
            state_t = state.available_edges.clone()
            action, prob = self.get_action_prob_from_state_and_heatmap(state, par_adv, select=select, temperature=temperature)
            state, reward, terminated, _, _ = self.env.step(action)

            actions.append(action.reshape(1))
            rewards.append(reward)
            cumulative_reward += reward.detach().reshape(1)
            # logits.append(logits_[action])
            probs.append(prob)
            available_edges.append(state_t)

        costs = float(self.env.current_cost.detach().cpu().numpy())

        self.cumulative_reward = cumulative_reward
        self.rewards = rewards
        # self.logits = logits
        self.probs = probs
        self.available_edges = available_edges
        self.actions = actions
        self.costs = costs
        if pruning:
            edge_list = self.env.edge_list
            edges = [tuple(edge_list[a1].squeeze().cpu().numpy()) for a1 in actions]
            new_graph = self.env.instance.edge_subgraph(edges).copy()
            new_graph.graph = self.env.instance.graph
            solution = remove_inessentials(new_graph, new_graph.graph['Terminals']['terminals'])

            c_huer = np.sum(list(map(lambda x: x[2]['cost'], solution.edges(data=True)))) / self.env.max_edge_cost
            costs = c_huer
            cumulative_reward = -c_huer

        return pack((cumulative_reward, rewards, probs, available_edges, actions, costs, ))

    def grad(self, par_shape, r_bl, cumulative_reward=None, rewards=None, probs=None, available_edges=None, actions=None, costs=None):
        cumulative_reward = cumulative_reward if cumulative_reward is not None else self.cumulative_reward
        rewards = rewards if rewards is not None else self.rewards
        # logits = logits if logits is not None else self.logits
        probs = probs if probs is not None else self.probs
        available_edges = available_edges if available_edges is not None else self.available_edges
        actions = actions if actions is not None else self.actions
        costs = costs if costs is not None else self.costs

        actions = torch.cat(actions, dim=0)

        grad = torch.zeros(size=par_shape)
        reward = cumulative_reward[-1] - r_bl

        grad[actions] -= reward
        for avail1, prob1 in zip(available_edges, probs):
            availables = avail1.nonzero(as_tuple=True)[0]
            grad[availables] += reward * prob1[availables]

        return grad

    @torch.no_grad()
    def get_action_prob_from_state_and_heatmap(self, state, heatmap, select='gumbel', temperature=1.0):
        par_e = torch.where(state.available_edges, heatmap, self.neg_inf)
        logits_ = par_e - torch.max(par_e)
        par_exp_ = (logits_ / temperature).exp()
        p_denom_ = torch.sum(par_exp_)
        probs_t = par_exp_ / p_denom_
        if select == 'max':
            action = torch.argmax(probs_t)
        elif select == 'gumbel':
            gumb = torch.empty_like(par_e).exponential_().log()
            action = torch.argmax(par_e - gumb)
        elif select == 'softmax':
            m = Categorical(probs_t)
            action = m.sample()
        else:
            raise ValueError
        return action, probs_t

    @classmethod
    def as_remote(cls, num_cpus=None, num_gpus=None, resources=None):
        return ray.remote(
            num_cpus=num_cpus, num_gpus=num_gpus, resources=resources)(cls)

    def multiple_rollout(self, jobs, status_worker):
        jobs = unpack_if_needed(jobs)
        results = []
        update_count = 0
        prev_update_time = datetime.datetime.now()
        for job in jobs:
            queue_idx = job['queue_idx']
            instance = job['data']
            select = job['select']
            temperature = job['temperature']
            pruning = job['pruning']
            initial_vertex_idx = job['initial_vertex_idx']
            start_time = datetime.datetime.now()

            state, info = self.env.reset(instance=instance, initial_vertex_idx=initial_vertex_idx)
            instance_emb, graph_emb, heatmap = get_heatmap_from_state(self.network, state, self.device)
            result = self.rollout(heatmap.detach(),
                                  instance=instance, select=select, temperature=temperature,
                                  pruning=pruning, initial_vertex_idx=initial_vertex_idx)

            elapsed = (datetime.datetime.now() - start_time).total_seconds()
            result = unpack_if_needed(result)
            _, _, _, _, _, cost = result
            results.append({
                'queue_idx': queue_idx,
                'cost': cost,
                'elapsed': elapsed
            })
            update_count += 1
            if (datetime.datetime.now() - prev_update_time).total_seconds() >= 0.1:
                prev_update_time = datetime.datetime.now()
                status_worker.update.remote(update_count)
                update_count = 0
        if update_count != 0:
            status_worker.update.remote(update_count)
        return pack(results)

    def get_heatmap_from_state(self, state):
        return get_heatmap_from_state(self.network, state)

@ray.remote(num_cpus=0.1)
class StatusWorker:
    def __init__(self, queue_len):
        self.queue_len = queue_len
        self.complete = 0
        self.diff = 0

    def update(self, num=1):
        self.complete += num
        self.diff += num

    def get_status(self):
        return self.complete

    def get_diff(self):
        diff = self.diff
        self.diff = 0
        return diff
