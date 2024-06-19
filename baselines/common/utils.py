from abc import ABC, abstractmethod
import gymnasium as gym
import networkx as nx
from typing import List, Dict, Union, Any
import numpy as np
from scipy.sparse.csgraph import dijkstra
# from stpgen.solvers.heuristic import get_complete, find_shortest_paths_from_nodes


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
    try:
        dist_matrix, predecessors = dijkstra(csgraph=graph, indices=[index_node[x] for x in nodes], limit=np.inf,
                                                min_only=False, directed=False, return_predecessors=True)
    except ValueError:
        dist_matrix, predecessors = dijkstra(csgraph=graph.todense(), indices=[index_node[x] for x in nodes], limit=np.inf,
                                                min_only=False, directed=False, return_predecessors=True)
        
    all_shortest_paths = {}  # Dictionary to store all pairs of shortest paths
    for i, source_node in enumerate(nodes):
        for _, destination_node in enumerate(nodelist):
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


class AbstractSTPEnv(ABC, gym.Env):
    def __init__(self, instance: nx.Graph=None):
        """STP environment should be started from a given STP instance.

        Args:
            instance (nx.Graph): STP instance.
        """
        super().__init__()
        self.instance = instance
        if self.instance:
            self.nterminals = instance.graph['Terminals']['meta']['numTerminals']
            self.terminals = instance.graph['Terminals']['terminals']
    
    @abstractmethod    
    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
    
    @abstractmethod
    def step(self, *args, **kwargs):
        super().step(*args, **kwargs)


class NodeFeature:
    
    @staticmethod
    def cherrypick(instance: nx.Graph, partial_solution: List[Any]=[], K: int=2, 
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
        
        # normalize edge cost within (0, 1)
        if instance.graph.get('Info') and normalize_edge_cost:
            max_cost = instance.graph.get('Info').get('max_edge_cost')
            if max_cost is None:
                max_cost = max_edge_cost
        else:
            max_cost = max_edge_cost
        
        node_sorted = sorted(instance.nodes)
        adj = nx.adjacency_matrix(instance, node_sorted, weight='cost')
        shortest_paths = find_shortest_paths_from_nodes(adj, terminals, node_sorted)
        
        features = {}
        for v in node_sorted:
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
        
    @staticmethod
    def cherrypick_agumented_coord(instance: nx.Graph, partial_solution: List[Any]=[], K: int=2, 
                                    normalize_distance=False) -> dict:
        features = NodeFeature.cherrypick(instance, partial_solution, K, normalize_distance)
        coordinator = GridCoordinate(instance)
        xs = np.linspace(0, 1, coordinator.dx)
        ys = np.linspace(0, 1, coordinator.dy)
        mesh = np.meshgrid(xs, ys, indexing='ij')
        
        for node, feat in features.items():
            i, j, k = coordinator.node_to_coordinate(node)
            x, y = mesh[0][i, j], mesh[1][i, j]
            faet_xy = np.array([x, y], dtype=np.float32)
            features.update({node: np.concatenate([feat, faet_xy])})
        
        return features
    
    
class GridCoordinate:
    def __init__(self, graph):
        self.graph = graph
        self.dx, self.dy, self.dz = graph.graph['Info']['dim'].values()
    
    def node_to_coordinate(self, node):
        z = node // (self.dx * self.dy)
        node = node % (self.dx * self.dy)
        y = node // self.dx
        x = node % self.dx
        return x, y, z
    
    def coordinate_to_node(self, x, y, z):
        assert 0 <= x < self.dx, 'invalid x value'
        assert 0 <= y < self.dy, 'invalid y value'
        assert 0 <= z < self.dz, 'invalid z value'
        return z * self.dx * self.dy + y * self.dx + x

if __name__ == "__main__":
    import sys
    sys.path.append('.')
    
    
    # class TestEnv(AbstractSTPEnv):
    #     def reset(self):
    #         print('ok')
            
    #     def step(self):
    #         print('ok')
    
    
    from stpgen.datasets.synthetic.instance import STPInstance_erdos_renyi
    from stpgen.datasets.synthetic.instance import STPInstance_grid
    instance_sampler = STPInstance_grid(n=20, p=1, seed=1234)
    inst = instance_sampler.sample()
    
    NodeFeature.cherrypick(inst)
    NodeFeature.cherrypick_agumented_coord(inst)
    print(1)
