import numpy as np
import copy
import torch

from baselines.cherrypick.env import STPEnvironment, get_max_edge_cost_from_state, STPState
from torch_scatter import segment_sum_csr, segment_max_csr, segment_min_csr
from baselines.common.utils import NodeFeature


class STPEdgeEnvironment(STPEnvironment):
    def __init__(self, instance_generator=None, seed=None, args=None, device='cpu') -> None:
        super().__init__(instance_generator, seed, args)
        self.current_cost = torch.zeros(1).to(device)
        self.edge_list = None
        self.edge_cost = None
        self.device = torch.device(device)

    def step(self, action):
        # action: edge_idx
        from1, to1 = self.edge_list[action]
        from1 = int(from1.cpu().numpy())
        to1 = int(to1.cpu().numpy())
        self.current_cost += self.edge_cost[action]

        self._check_action_validity(to1)
        self.cost_prev = self.state.cost
        self._transision(to1, current_edge=action)

        state = copy.deepcopy(self.state)
        terminated = self._is_done()
        reward = self._get_reward()
        truncated = False
        info = None
        return state, reward, terminated, truncated, info

    def _get_current_cost(self, return_float=False):
        if return_float:
            return float(self.current_cost.detach().cpu().numpy())
        return self.current_cost

    def _get_cost_from_edge_idxes(self, n1, n2, normalize=True):
        if n2 < n1:
            n1, n2 = n2, n1
        max_cost = self.max_edge_cost if normalize else 1

        return self.instance.adj[n1][n2]['cost'] / max_cost

    def reset(self, instance=None, initial_vertex_idx=None):
        self.current_cost = torch.zeros(1).to(self.device)
        self._init_instance(instance)
        if initial_vertex_idx is None:
            initial_vertex = self.rng.choice(self.terminals, 1)[0]
        else:
            initial_vertex = self.terminals[initial_vertex_idx]
        n_nodes = len(self.instance.nodes)
        edge_list = list(self.instance.edges)
        edge_list_reversed = [e1[::-1] for e1 in edge_list]
        edge_list += edge_list_reversed
        edge_list = np.array(edge_list)
        edge_cost = np.array([d1[2]['cost'] / self.max_edge_cost for d1 in list(self.instance.edges(data=True))] * 2).astype(np.float32)

        sorted_idx = np.argsort(np.apply_along_axis(lambda r: r[0] * n_nodes + r[1], 1, edge_list), axis=0)
        self.edge_list = torch.from_numpy(edge_list[sorted_idx]).to(self.device)
        self.edge_cost = torch.from_numpy(edge_cost[sorted_idx]).to(self.device)

        self.state.edge_list = self.edge_list
        self.state.edge_cost = self.edge_cost
        self.state.available_edges = torch.from_numpy(np.array([False for _ in range(len(edge_list))], dtype=bool)).to(self.device)

        self.start_position = [None for _ in range(n_nodes)]
        cur_nodes = -1

        for idx in range(len(edge_list)):
            if self.edge_list[idx][0] != cur_nodes:
                cur_nodes = self.edge_list[idx][0]
                self.start_position[cur_nodes] = idx
        self.start_position.append(len(edge_list))

        self._transision(initial_vertex)
        self.cost_prev = 0

        state = self.state
        info = None

        return state, info

    def _transision(self, vertex, current_edge=None):
        super()._transision(vertex)

        # make False to current edge
        if current_edge is not None:  # in case of initial selection
            self.state.available_edges[current_edge] = False

        # make False to edges of which the target is the vertex
        self.state.available_edges[self.edge_list[:, 1] == vertex] = False

        # make True to edges of which the source is the vertex and the target is not in partial solution
        partial = torch.IntTensor(self.state.partial_solution).to(self.device)
        all_idx = torch.arange(start=self.start_position[vertex], end=self.start_position[vertex + 1], device=self.device)
        edge_filter = ~torch.isin(self.edge_list[all_idx][:, 1], partial)
        edge_idx = all_idx[edge_filter]
        self.state.available_edges[edge_idx] = True

    def _get_reward(self):
        terminated = self._is_done()
        if not terminated:
            return torch.zeros_like(self.current_cost).to(self.device)
        else:
            return -self.current_cost


class STPTorchEnvironment(STPEnvironment):
    def __init__(self, instance_generator=None, seed=None, device='cpu') -> None:
        super().__init__(instance_generator, seed)
        self.device = torch.device(device)

    def _init_instance(self, instance=None,  num_instance=100):
        self.num_instance = num_instance
        if instance is not None:
            self.instance = [instance.copy() for _ in range(num_instance)]
        else:
            self.instance = [self.instance_generator.sample() for _ in range(num_instance)]

        self.nterminals = torch.IntTensor([i1.graph['Terminals']['meta']['numTerminals'] for i1 in self.instance]).to(self.device)
        self.nterminals_cumsum = torch.empty(num_instance + 1, dtype=torch.int64, device=self.device)
        self.nterminals_cumsum[0] = 0
        self.nterminals_cumsum[1:] = self.nterminals.cumsum(0)
        self.total_terminals = self.nterminals_cumsum[-1].cpu().item()
        self.terminals = torch.cat([torch.from_numpy(i1.graph['Terminals']['terminals']).to(self.device) for i1 in self.instance], dim=0).to(self.device)
        self.state = [STPState(graph=i1) for i1 in self.instance]
        self.max_edge_cost = get_max_edge_cost_from_state(self.state[0])
        self.n_nodes = torch.IntTensor([len(i1.nodes) for i1 in self.instance]).to(self.device)
        n_nodes_np = self.n_nodes.cpu().numpy()
        edge_list = [list(i1.edges) for i1 in self.instance]
        edge_list_reversed = [None for _ in range(num_instance)]
        for idx in range(num_instance):
            edge_list_reversed[idx] = [e1[::-1] for e1 in edge_list[idx]]

        edge_list = [e1 + e2 for e1, e2 in zip(edge_list, edge_list_reversed)]
        edge_list = [torch.IntTensor(e1) for e1 in edge_list]
        edge_cost = [torch.FloatTensor([d1[2]['cost'] / self.max_edge_cost for d1 in list(i1.edges(data=True))] * 2).to(self.device) for i1 in self.instance]
        sorted_idx = [np.argsort(np.apply_along_axis(lambda r: r[0] * n_nodes_np[i1] + r[1], 1, edge_list[i1]), axis=0) for i1 in range(len(edge_list))]
        edge_list = [edge_list[i1][sorted_idx[i1]] for i1 in range(num_instance)]
        edge_cost = [edge_cost[i1][sorted_idx[i1]] for i1 in range(num_instance)]
        graph_idx = [torch.full((len(edge_list[idx]), ), idx) for idx in range(num_instance)]

        self.graph_idx = torch.cat(graph_idx, dim=0).to(self.device)
        self.edge_list = torch.cat(edge_list, dim=0).to(self.device)
        self.edge_cost = torch.cat(edge_cost, dim=0).to(self.device)
        self.edge_num = torch.IntTensor([e1.shape[0] for e1 in edge_list]).to(self.device)
        self.edge_num_cumsum = torch.empty(num_instance + 1, dtype=torch.int64, device=self.device)
        self.edge_num_cumsum[0] = 0
        self.edge_num_cumsum[1:] = self.edge_num.cumsum(0)

        self.nodes_cumsum = torch.empty(num_instance + 1, dtype=torch.int64, device=self.device)
        self.nodes_cumsum[0] = 0
        self.nodes_cumsum[1:] = self.n_nodes.cumsum(0)

        degrees = [torch.from_numpy(np.array(i1.degree)[:, 1]).to(self.device) for i1 in self.instance]
        self.degrees = torch.cat(degrees, dim=0)
        self.degrees_cumsum = torch.empty(self.nodes_cumsum[-1] + 1, dtype=torch.int64, device=self.device)
        self.degrees_cumsum[0] = 0
        self.degrees_cumsum[1:] = self.degrees.cumsum(0)

        self.partial_solution = torch.zeros((self.nodes_cumsum[-1], ), dtype=torch.bool).to(self.device)
        self.available_edges = torch.zeros_like(self.edge_cost, dtype=torch.bool).to(self.device)
        self.current_cost = torch.zeros((num_instance, ), dtype=torch.float32).to(self.device)

    def _is_done(self):
        graph_idx = torch.arange(self.num_instance)
        graph_idx = graph_idx.repeat_interleave(self.nterminals)
        offset = self.nodes_cumsum[graph_idx]
        terminal = offset + self.terminals
        terminal_included = (self.partial_solution[terminal]).float()
        is_done = segment_min_csr(terminal_included, self.nterminals_cumsum)[0].bool()

        return is_done

    def _get_reward(self):
        terminated = self._is_done()
        return torch.where(terminated, -self.current_cost, torch.zeros_like(self.current_cost, device=self.device))

    def _get_current_cost(self, return_float=False):
        if return_float:
            return self.current_cost.detach().cpu().numpy().tolist()
        return self.current_cost

    def _transision(self, vertex, edge=None):
        self.partial_solution[vertex + self.nodes_cumsum[:-1]] = True
        if edge is not None:
            self.available_edges[edge + self.edge_num_cumsum[:-1]] = False
            # from_vertex, _ = self.edge_list[edge + self.edge_num_cumsum[:-1]]
        partial_slice = [self.partial_solution[n1:n2] for n1, n2 in zip(self.nodes_cumsum[:-1], self.nodes_cumsum[1:])]
        edge_slice = [self.edge_list[n1:n2] for n1, n2 in zip(self.edge_num_cumsum[:-1], self.edge_num_cumsum[1:])]
        destination_filter = torch.cat([s1[:, 1] == v1 for s1, v1 in zip(edge_slice, vertex)], dim=0)

        self.available_edges[destination_filter] = False

        departure_filter = torch.cat([s1[:, 0] == v1 for s1, v1 in zip(edge_slice, vertex)], dim=0)
        partial_filter = torch.cat([~torch.isin(s1[:, 1], p1.nonzero()) for s1, p1 in zip(edge_slice, partial_slice)], dim=0)
        departure_filter = torch.logical_and(departure_filter, partial_filter)

        self.available_edges[departure_filter] = True

        for idx, (e1, e2, n1, n2) in enumerate(zip(self.edge_num_cumsum[:-1], self.edge_num_cumsum[1:],
                                           self.nodes_cumsum[:-1], self.nodes_cumsum[1:])):
            self.state[idx].edge_list = self.edge_list[e1:e2]
            self.state[idx].edge_cost = self.edge_cost[e1:e2]
            self.state[idx].available_edges = self.available_edges[e1:e2]
            self.state[idx].partial_solution = self.partial_solution[n1:n2].nonzero().squeeze(1)
            if vertex[idx].cpu().item() in self.instance[idx].graph['Terminals']['terminals']:
                self.state[idx].distance = NodeFeature.cherrypick(self.state[idx].graph, self.state[idx].partial_solution,
                                                        K=2, normalize_edge_cost=True)
            self.state[idx].cost = self.current_cost[idx]
            self.state[idx].step += 1

    def reset(self, instance=None, initial_vertex_idx=None, num_instance=100):
        self._init_instance(instance, num_instance=num_instance)
        if initial_vertex_idx is None:
            rnd_v = torch.rand(size=(self.total_terminals, ), dtype=torch.float32).to(self.device)
            value, initial_vertex_idx = segment_max_csr(rnd_v, self.nterminals_cumsum)
        else:
            initial_vertex_idx += self.nterminals_cumsum[:-1]
        initial_vertex = self.terminals[initial_vertex_idx]
        self._transision(initial_vertex)
        return self.state, None

    def step(self, action):
        edge_idx = action + self.edge_num_cumsum[:-1]
        available = self.available_edges[edge_idx]
        if not torch.all(available):
            raise ValueError('Invalid action!')
        edges = self.edge_list[edge_idx]
        from1 = edges[:, 0]
        to1 = edges[:, 1]
        self.current_cost += self.edge_cost[edge_idx]
        self._transision(to1, edge=action)

        state = copy.deepcopy(self.state)
        terminated = self._is_done()
        reward = self._get_reward()

        return state, reward, terminated, [False for _ in range(self.num_instance)], None


if __name__ == '__main__':
    from stpgen.datasets.synthetic.instance import (MAX_EDGE_COST,
                                                     STPInstance_erdos_renyi,
                                                     STPInstance_grid,
                                                     STPInstance_regular,
                                                     STPInstance_watts_strogatz)

    instance_sampler = STPInstance_erdos_renyi(n=20, seed=1234, cost_type='gaussian')
    env = STPTorchEnvironment(instance_sampler, 1234)
    state,_ = env.reset(num_instance=10)
    action = []
    for s1 in state:
        idx = s1.available_edges.nonzero().squeeze(1)
        perm = torch.randperm(idx.size(0))
        action.append(idx[perm[0]].reshape(1))
    action = torch.cat(action, dim=0)
    result = env.step(action)
    print('hi')


