from abc import ABC, abstractmethod
# import gymnasium as gym
import networkx as nx
from typing import List, Dict, Union, Any
from scipy.sparse.csgraph import dijkstra
import numpy as np
import torch
import heapq
import pdb
import math
from itertools import chain, combinations
import networkx as nx
import time
import torch.nn.functional as F
from torch.distributions import Categorical



class NodeFeature:
    
    @staticmethod
    def cherrypick(instance: nx.Graph, partial_solution: List[Any]=[], K: int=2,
                    # normalize_distance=True, normalize_edge_cost=False, max_edge_cost=10) -> dict:
                    normalize_edge_cost=False, max_edge_cost=1.) -> dict:
        """
        get distance-based node feature, which calculate distance 
        from each node to K-nearest terminals.
        
        Args:
            instance (nx.Graph): The input graph.
            partial_solution (List[Any], optional): If terminals already in partial solution, 
                                                    distance calculation will exclude them. 
                                                    Defaults to [].
            K (int, optional): The number of nearest terminals to consider. Defaults to 2.
        
        Returns:
            dict: A dictionary of node features, where the keys are nodes and the values are arrays 
                representing the distances to the K-nearest terminals.
        """
        terminals = instance.graph['Terminals']['terminals']
        node_sorted = sorted(instance.nodes)
        adj = nx.adjacency_matrix(instance, node_sorted, weight='cost')
        max_cost = adj.max()
        shortest_paths = find_shortest_paths_from_nodes(adj, terminals, node_sorted)

        features = {}
        for v in sorted(instance.nodes):
            feature = np.zeros(K)
            d = sorted([shortest_paths.get((t, v), (0, ))[0] / max_cost for t in terminals if t not in partial_solution])

            if len(d) >= K:
                feature = np.array(d[:K])
            else:
                feature[:len(d)] = d

            features[v] = feature
        return features


    @staticmethod
    def binary(instance: nx.Graph, partial_solution: List[Any]=[]):
        """
        Generate binary features for each node in the graph. A binary feature 
        vector includes presence of neighbors, whether the node is a terminal 
        and whether it is part of a partial solution.

        Args:
            instance (nx.Graph): The input graph.
            partial_solution (List[Any], optional): List of nodes that are part 
                                                    of a partial solution. Defaults to [].
        
        Returns:
            dict: A dictionary containing node-feature pairs. The feature is a 
                binary vector where the first N elements indicate the presence of 
                neighbor nodes (1 for presence, 0 for absence), the second last 
                element indicates whether the node is a terminal, and the last 
                element indicates whether the node is part of the partial solution.
        """
        terminals = instance.graph['Terminals']['terminals']
        num_nodes = instance.number_of_nodes()
        
        features = {}
        for v in instance.nodes:
            feature = np.zeros(num_nodes + 2) # (is_terminal, is_partial_solution)
            neighbors = list(instance.neighbors(v))
            feature[neighbors] = 1
            
            if v in terminals:
                feature[-2] = 1
            
            if v in partial_solution:    
                feature[-1] = 1
            
            features[v] = feature
        return features
        

def get_numpy_from_nonfixed_2d_array(aa, fixed_length, padding_value=0):
    rows = []
    for a in aa:
        rows.append(np.pad(list(chain(*a)), (0, fixed_length), 'constant', constant_values=padding_value)[:fixed_length])
    return np.concatenate(rows, axis=0).reshape(-1, fixed_length)
    
def flatten_2d_array(aa):
    return np.array([list(chain(*a))+ [-1] for a in aa] )   # -1 for EOS

def find(parent, i):
    if parent[i] == i:
        return i 
    return find(parent, parent[i])

def union(parent, rank, x, y):
    root_x = find(parent, x)
    root_y = find(parent, y)

    if rank[root_x] < rank[root_y]:
        parent[root_x] = root_y
    elif rank[root_x] > rank[root_y]:
        parent[root_y] = root_x
    else:
        parent[root_y] = root_x
        rank[root_x] += 1


def stp_merge_tours(adj_matrix, terminals, edge_cost):
    # 2-approx algorithm
    N = np_points.shape[0]
    terminal_nodes = terminals[terminals != -1].tolist()
    # make graph
    G_masking = nx.Graph()
    
    masking_max_weight = np.max(adj_matrix) + 0.01
    for i in range(N):
        for j in range(N):
            if edge_cost[0, i, j] != 0:
                G_masking.add_edge(i, j, weight=masking_max_weight-adj_matrix[0,i,j].item())

    # G_origin_subgraph = nx.Graph()
    G_masking_subgraph = nx.Graph()

    for u in terminal_nodes:
        for v in terminal_nodes:
            if u!= v:
                if nx.has_path(G_masking, u, v):
                    path = nx.shortest_path(G_masking, u, v, weight='weight')
                    nx.add_path(G_masking_subgraph, path)
                else:
                    raise ValueError("G_masking has no shortest path between Terminals")
    
    G_masking_mst = nx.minimum_spanning_tree(G_masking_subgraph)

    # mapping mst to Graph
    G_masking_steiner_tree = nx.Graph()

   
    for u,v in G_masking_mst.edges():
        path = nx.shortest_path(G_masking, u, v, weight='weight')
        nx.add_path(G_masking_steiner_tree, path)
    
  
    for node in list(G_masking_steiner_tree.nodes()):
        if node not in terminal_nodes and G_masking_steiner_tree.degree(node) == 1:
            G_masking_steiner_tree.remove_node(node)
    G_masking_edge_list = list(G_masking_steiner_tree.edges())
    
    # return G_origin_edge_list, G_masking_edge_list
    return G_masking_edge_list



class InfeasibleProblemError(Exception):
    def __init__(self, msg=""):
        self.msg = msg
    
    def __str_(self):
        return f"{self.__class__.__name__}: {self.msg}"

class TwoApproximation:
    def __init__(self, adj_matrix, terminals, edge_cost, alpha=0.9) -> None:
        self.adj_matrix = adj_matrix
        self.terminals = terminals[terminals != -1].tolist()
        self.edge_cost = edge_cost.cpu().numpy()
        self.alpha = alpha
        self.solution_masked = None
        self.solution_unmasked = None
        self.solution_edge_cost = None
        self.solution_none_cost = None

    def solve(self, verbose=False):
        norm_edge_cost = self.edge_cost / self.edge_cost.max()

        # Create masking
        edge_mask = (self.edge_cost != 0).astype(float)

        masked_adj_matrix = self.adj_matrix * edge_mask
        masked_norm_edge_cost = norm_edge_cost * edge_mask

        combined_adj_matrix_masked = self.alpha * (1-masked_adj_matrix) + (1-self.alpha) * masked_norm_edge_cost
        combined_adj_matrix_unmasked = self.alpha * (1-self.adj_matrix) + (1-self.alpha) * norm_edge_cost
        adj_matrix_edge_cost = norm_edge_cost
        adj_matrix_none_cost = 1-self.adj_matrix

        G_masked = self.create_graph(combined_adj_matrix_masked, edge_mask)
        G_unmasked = self.create_graph(combined_adj_matrix_unmasked)
        G_edge_cost = self.create_graph(adj_matrix_edge_cost, edge_mask)
        G_none_cost = self.create_graph(adj_matrix_none_cost, edge_mask)

        self.solution_masked = TwoApprox(G_masked, self.terminals, weight='cost')
        self.solution_unmasked = TwoApprox(G_unmasked, self.terminals, weight='cost')
        self.solution_edge_cost = TwoApprox(G_edge_cost, self.terminals, weight='cost')
        self.solution_none_cost = TwoApprox(G_none_cost, self.terminals, weight='cost')

    def create_graph(self, adj_matrix, edge_mask=None):
        G = nx.Graph()
        G.add_nodes_from(range(adj_matrix.shape[1]))

        for i in range(adj_matrix.shape[1]):
            for j in range(i+1, adj_matrix.shape[2]):
                if edge_mask is None or edge_mask[0, i, j] != 0:
                    G.add_edge(i, j, cost=adj_matrix[0, i, j].item())

        return G

def TwoApprox(graph, terminals, weight='cost', edge_cost=None):
    '''
    find a single Steiner tree.

    Step 1: compute the complete undirected subgraph G1 of terminal nodes in original graph G, where the complete graph is fully-connected graph in which each edge is weighted by the shortest path distance between the nodes.
    Step 2: find the minimum spanning tree T1 of G1. (remove redundant edge of virtual graph which consists of terminal nodes)
    Step 3: construct the subgraph Gs which replace each edge in T1 by shortest path of G.
    Step 4: find the minimum spanning tree Ts of Gs. (remove cyclic edges of Gs)
    Step 5: make sure all the leaves in Ts are terminal nodes by deleting edges in Ts, if necessary.

    References
    ------------    
    Kou, Lawrence, George Markowsky, and Leonard Berman. "A fast algorithm for Steiner trees." Acta informatica 15.2 (1981): 141-145.
    '''

    m = nx.adjacency_matrix(graph, nodelist=sorted(graph.nodes), weight='cost')
    terminal_complete = get_complete(m, terminals, nodelist=sorted(graph.nodes))
    terminal_mst = find_mst(terminal_complete, 'weight')
    terminal_complete.edges(data=True)
    cand_subgraph = restore_graph(graph, terminal_mst)
    subgraph = find_mst(cand_subgraph, weight)
    ST = remove_inessentials(subgraph, terminals)

    return ST


def get_complete(given_g, selected_nodes, nodelist=None):
    """
    construct complete graph of selected nodes of which edge weight is shortest path length.

    Parameters
    -------------
    given_g : nx.Graph(). given graph
    selected_nodes : list of nodes

    Returns
    ------------
    complete_g : nx.Graph(). It has two attributes which are 'shortest-path' and "weight".
                    'shortest-path' is a sequential list of nodes.
                    "weight" means shortest path length.
    """
    complete_g = nx.complete_graph(selected_nodes)
    all_shortest_paths = find_shortest_paths_from_nodes(given_g, selected_nodes, nodelist)
    
    attrs = {}

    for ti, tj in combinations(selected_nodes, 2):
        c, path = all_shortest_paths[(ti, tj)]

        if c == math.inf:
            raise InfeasibleProblemError('infeasible solution')

        attrs[(ti, tj)] = {
            'shortest-path' : path,
            'weight' : c,
        }
    nx.set_edge_attributes(complete_g, attrs)
    return complete_g


def find_shortest_paths_from_nodes(graph, nodes: list, nodelist=None):
    """_summary_

    Args:
        graph (sparse matrix / array): weighted adjacency matrix 
        nodes (list): terminal nodes
        nodelist (_type_, optional): list of all nodes ordered by adjacency matrix

    Returns:
        shortest path for all terminal pairs 
    """
    index_node = {n: i for i, n in enumerate(nodelist)}

    dist_matrix, predecessors = dijkstra(csgraph=graph.todense(),
                                             indices=[index_node[x] for x in nodes],
                                             limit=np.inf,min_only=False, directed=False,
                                             return_predecessors=True)
    all_shortest_paths = {} # Dictionary to store all pairs of shortest paths
    for i, source_node in enumerate(nodes):
        for _, destination_node in enumerate(nodes):
            # Skip paths from a node to itself
            if source_node == destination_node: continue  
            # Skip paths for infinity cost
            destination_idx = index_node[destination_node]
            dij = dist_matrix[i, destination_idx]
            
            if dij is np.inf: continue

            # Initialize the path list with the destination node
            path = [destination_node]

            # Trace back from the destination node to the source node using predecessors
            while path[-1] != source_node:
                current_node = path[-1]
                predecessor_idx = predecessors[i, index_node[current_node]]
                # print(dist_matrix[i, current_node])

                if predecessor_idx == -9999:
                    # No path exists
                    break
                predecessor_node = nodelist[predecessor_idx]
                path.append(predecessor_node)

            # Reverse the path to get it from source to destination
            path.reverse()

            # Store the path in the dictionary
            all_shortest_paths[(source_node, destination_node)] = (dij, path)
    return all_shortest_paths

def find_mst(g, weight):
    return nx.minimum_spanning_tree(g, weight=weight)

def restore_graph(given_g, virtual_g): 
    """230529: New Version
    """
    g = nx.Graph()

    def extract_edges(nodes):  # extract edge (node pair) from path (sequential node list).
        return [(nodes[ix - 1], nodes[ix]) for ix in range(1, len(nodes))]

    for node_path in nx.get_edge_attributes(virtual_g, "shortest-path").values():
        edges = extract_edges(node_path)
        nodes_list = []
        edges_list = []
        for edge in edges:
            nodes_list.append(edge[0])
            edges_list.append((edge[0], edge[1]))
        nodes_list.append(edge[1])
        g.add_nodes_from(nodes_list)
        g.add_edges_from(edges_list)

    return g

def remove_inessentials(cand_g, terminals):
    """
    remove redundant nodes and edges to ensure that all leaf nodes are terminal nodes.

    Parameters
    ------------
    cand_g : nx.Graph(). Steiner Tree candidate graph.
    terminals : list of tuples. It contains terminal nodes.
    """
    leafnodes = [n for n, dg in cand_g.degree() if dg == 1]
    not_terminals = leafnodes
    while not_terminals:
        node = not_terminals.pop()
        if node in terminals:
            continue
        # if leaf node is not terminal, this node will be removed.
        neighbors = list(cand_g.neighbors(node))
        assert len(neighbors) == 1  # degree of leaf node is 1
        while neighbors:
            nei = neighbors.pop()
            cand_g.remove_edge(node, nei)
            # again not terminal leaf node
            if (cand_g.degree[nei] == 1) and (nei not in terminals):
                not_terminals.append(nei)
        cand_g.remove_node(node)
    return cand_g


def calculate_cost(tree_edges, edge_cost):
    total_cost = 0
    for u, v in tree_edges:
        if edge_cost[0, u, v] != 0:
            total_cost += edge_cost[0, u, v].item()
        else:
            total_cost += 10000
        # if (u, v) in edge_cost:
        #     total_cost += edge_cost[(u,v)].item()
        # else:
        #     total_cost += math.inf

    return total_cost

def remove_duplicate(tree_edges):
    unique_edges = set()
    for u, v in tree_edges:
        if (u,v) not in unique_edges and (v, u) not in unique_edges:
            unique_edges.add((min(u,v), max(u,v)))
    return list(unique_edges)


def two_opt_stp(tree_edges, edge_cost, adj_matrix, terminals):
    N = adj_matrix.shape[1]
    improved = True

    while improved:
        improved = False
        for i in range(len(tree_edges)):
            for j in range(i+1, len(tree_edges)):
                u1, v1 = tree_edges[i]
                u2, v2 = tree_edges[j]
                # if (u1, v2) in edge_cost and (u2, v1) in edge_cost:
                if edge_cost[0, u1, v2] != 0 and edge_cost[0, u2, v1] !=0:
                    new_edges = [edge for k, edge in enumerate(tree_edges) if k != i and k != j] + [(u1, v2), (u2, v1)]
                    new_cost = calculate_cost(new_edges, edge_cost)
                    current_cost = calculate_cost(tree_edges, edge_cost)

                    if new_cost < current_cost:
                        tree_edges = new_edges
                        improved = True
                
                else:
                    alternatives = []
                    for u in [u1, v1]:
                        for v in [u2, v2]:
                            # if (u,v) in edge_cost:
                            if edge_cost[0, u, v] != 0:
                                alternatives.append((u,v))

                    if len(alternatives) == 2:
                        new_edges = [edge for k, edge in enumerate(tree_edges) if k != i and k != j] + alternatives
                        new_cost = calculate_cost(new_edges, edge_cost)
                        current_cost = calculate_cost(tree_edges, edge_cost)

                        if new_cost < current_cost:
                            tree_edges = new_edges
                            improved = True
    tree_edges = remove_duplicate(tree_edges)
    return tree_edges


class Cherrypick:
    def __init__(self, graph, terminals, edge_cost, alpha):
        self.graph = graph
        self.terminals = terminals[terminals != -1].tolist()
        self.edge_cost = edge_cost.cpu().numpy()
        self.alpha = alpha
        self.solution_none_cost = None
        self.none_cost_edge_available = np.full((graph.shape[1], graph.shape[1]), False, dtype=bool)

    def solve(self, verbose=False):
        norm_edge_cost = 1 - (self.edge_cost / self.edge_cost.max())
        masked_adj = self.alpha * self.graph + (1-self.alpha) * norm_edge_cost

        # select terminal and init partial solution
        G_none_cost = self.create_graph(self.graph, self.edge_cost)

        # partial = set(np.random.choice(self.terminals, 1, replace=False))
        
        # partial = set(np.array([self.terminals[0]]))
        none_cost_partial = self.select_max_weight_edges(G_none_cost, self.terminals)
        
        self.solution_none_cost = self.pick(G_none_cost, none_cost_partial, self.none_cost_edge_available)
        
        self.solution_none_cost = remove_inessentials(nx.Graph(self.solution_none_cost), self.terminals)



    def create_graph(self, adj_matrix, edge_mask=None):
        G = nx.Graph()
        G.add_nodes_from(range(adj_matrix.shape[1]))

        for i in range(adj_matrix.shape[1]):
            for j in range(i+1, adj_matrix.shape[2]):
                if edge_mask is None or edge_mask[0, i, j] != 0:
                    G.add_edge(i, j, cost=adj_matrix[0, i, j].item())

        return G
    
    def select_max_weight_edges(self, G, terminals):
        edges = []
        for terminal in terminals:
            edges.extend(G.edges(terminal, data='cost'))
        
        max_edge = max(edges, key=lambda x : x[2])
        return max_edge[:2]


    def pick(self, graph, partial, available_edge):
        
        weight_matrix = nx.to_numpy_array(graph, weight='cost')
        selected_edge = [partial]
        current_nodes = set(partial)
        # while all terminal node is in parital solution
        while set(self.terminals) - current_nodes:
            # Get boundary edges
            boundary_edges = list(nx.edge_boundary(graph, current_nodes))

            # action(edge) select 
            available_edge[tuple(zip(*boundary_edges))] = True
            logits_ = np.where(available_edge, weight_matrix, -np.inf)

            # argmax
            row_idx, col_idx = np.unravel_index(np.argmax(logits_), logits_.shape)

            if not row_idx in current_nodes and not col_idx in current_nodes:
                raise ValueError('selected edge not exist in partial')

            # add partial solution (node)
            current_nodes = current_nodes | {row_idx, col_idx}
            selected_edge.append((row_idx, col_idx))
            available_edge[:] = False
        
        return selected_edge


class STPEvaluator():
    def __init__(self):
        pass

    def evaluate(self, tours, edge_cost):
        total_cost = 0
        for e in tours:
            if e[0] == -1 or e[1] == -1:
                raise ValueError("Evaluate has -1 node")
            if edge_cost[0, e[0], e[1]] != 0:
                total_cost += edge_cost[0, e[0], e[1]].item()
            else:
                raise ValueError('model select unavailable edge')
                total_cost += edge_cost.max().item()
        
        return total_cost

