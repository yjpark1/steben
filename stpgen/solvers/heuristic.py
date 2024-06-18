import math
from itertools import combinations

import networkx as nx
import numpy as np
from scipy.sparse.csgraph import dijkstra
import time



class InfeasibleProblemError(Exception):
    def __init__(self, msg=""):
        self.msg = msg

    def __str__(self):
        return f"{self.__class__.__name__}: {self.msg}"
    

class TwoApproximation:
    def __init__(self, graph) -> None:
        self.graph = graph
        self.solution = None
        self.duration = None
    
    def solve(self, verbose=False):
        t0 = time.time()
        self.solution = TwoApprox(self.graph, sorted(self.graph.graph['Terminals']['terminals']), 
                                  weight='cost')
        self.duration = time.time() - t0
        

def TwoApprox(graph, terminals, weight='cost'):
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
    subgraph = find_mst(cand_subgraph, 'cost')
    ST = remove_inessentials(subgraph, terminals)
    
    return ST


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


def get_complete(given_g, selected_nodes, nodelist=None):
    """
    construct complete graph of selected nodes of which edge weight is shortest path length.

    Parameters
    -------------
    given_g : nx.Graph(). given graph
    selected_nodes : list of nodes

    Returns
    ------------
    complete_g : nx.Graph(). It has two attributes which are 'shortest-path' and "cost".
                    'shortest-path' is a sequential list of nodes.
                    "cost" means shortest path length.
    """

    complete_g = nx.complete_graph(selected_nodes)
    all_shortest_paths = find_shortest_paths_from_nodes(given_g, selected_nodes, nodelist)
    
    attrs = {}
    for ti, tj in combinations(selected_nodes, 2):
        c, path = all_shortest_paths[(ti, tj)]
        
        if c == math.inf:
            raise InfeasibleProblemError('infeasible solution')
        
        attrs[(ti, tj)] = {
            "shortest-path": path,
            "weight": c,
        }
    nx.set_edge_attributes(complete_g, attrs)
    return complete_g


def get_subgraph_nodes(nodes_G_1d, bounds):
    """Get the possible subgraph nodes under the bounds (from min_bound to max_bound)
    nodes = [(1,2,3), (4,5,6), ..., (10, 11, 12)]
    bounds = (min_bound, max_bound)
    min_bound = (x_min, y_min, z_min), 
    max_bound = (x_max, y_max, z_max)
    """
    min_bound, max_bound = bounds
    # print('min_bound',min_bound, 'max_bound',max_bound)
    # print('nodes_G_1d',nodes_G_1d)
    nodes_sub_1d = nodes_G_1d[min_bound[0]:max_bound[0], min_bound[1]:max_bound[1], min_bound[2]:max_bound[2]]
    nodes_sub_1d = set(nodes_sub_1d.flatten()) - {-1}
    nodes_sub_1d = np.array(list(nodes_sub_1d))  

    return nodes_sub_1d


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