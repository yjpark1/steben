import torch
import numpy as np
import networkx as nx

from stpgen.datasets.synthetic.instance import (MAX_EDGE_COST,
                                                 STPInstance_erdos_renyi,
                                                 STPInstance_grid,
                                                 STPInstance_regular,
                                                 STPInstance_watts_strogatz)

from baselines.cherrypick.env import STPEnvironment
from baselines.dimes.env import STPEdgeEnvironment


def get_instance_sampler_from_graph_type(n_nodes, seed, graph_type, cost_type):
    if graph_type == "ER":
        instance_sampler = STPInstance_erdos_renyi(n=n_nodes, seed=seed, cost_type=cost_type)
    elif graph_type == "Grid":
        instance_sampler = STPInstance_grid(n=n_nodes, seed=seed, cost_type=cost_type)
    elif graph_type == "RR":
        instance_sampler = STPInstance_regular(n=n_nodes, seed=seed, cost_type=cost_type)
    elif graph_type == "WS":
        instance_sampler = STPInstance_watts_strogatz(n=n_nodes, seed=seed, cost_type=cost_type)
    else:
        raise ValueError
    return instance_sampler


def new_env(n_nodes, seed, args, graph_type='ER', cost_type='gaussian', device=torch.device('cpu')):
    instance_sampler = get_instance_sampler_from_graph_type(n_nodes, seed, graph_type, cost_type)
    return STPEdgeEnvironment(instance_sampler, seed=seed, device=device, args=args)



def create_binary_vector(indices, size):
    binary_vector = np.zeros(size, dtype=np.float32)
    # Set the specified indices to 1
    binary_vector[indices] = 1
    return binary_vector


def process_single_state(state, device):
    num_vertices = state.graph.number_of_nodes()
    sv = create_binary_vector(state.partial_solution, num_vertices)
    tv = create_binary_vector(state.graph.graph['Terminals']['terminals'], num_vertices)
    xv = np.array([v for _, v in sorted(state.distance.items(), key=lambda x: x[0])])
    adj = nx.adjacency_matrix(state.graph).todense()

    sv = np.array([sv], dtype=np.float32)
    tv = np.array([tv], dtype=np.float32)
    xv = np.array([xv], dtype=np.float32)
    adj = np.array([adj], dtype=np.float32)

    features = (
        torch.from_numpy(sv).to(device),
        torch.from_numpy(tv).to(device),
        torch.from_numpy(xv).to(device),
        torch.from_numpy(adj).to(device)
    )

    return features


def state_to_emb(network, states, reduce_dim=False, device=torch.device('cpu')):
    if not isinstance(states, list):
        states = [states]

    instance_emb = []
    graph_emb = []
    for state in states:
        s, t, x, adj = process_single_state(state, device)
        emb = network.graph_emb(s, t, x).squeeze(0)
        instance_emb.append(emb)

        emb2 = network.emb_net(emb, state.edge_list.transpose(0, 1), state.edge_cost.unsqueeze(-1))
        graph_emb.append(emb2)

    if reduce_dim and len(states) == 1:
        return instance_emb[0], graph_emb[0]

    return instance_emb, graph_emb

@torch.no_grad()
def get_heatmap_from_state(network, state, device=torch.device('cpu')):
    instance_emb, graph_emb = state_to_emb(network, state, reduce_dim=True, device=device)
    heatmap = network.par_net(graph_emb)
    return instance_emb, graph_emb, heatmap