from typing import Any
import networkx as nx
import numpy as np
import abc
from scipy.sparse import lil_matrix, csr_matrix
import types
import re


# MAX_EDGE_COST = 2**16
MAX_EDGE_COST = 1.


class AbstractInstance(metaclass=abc.ABCMeta):
    """Generate single instance of Steiner tree problem.

    Args:
        metaclass (_type_, optional): _description_. Defaults to abc.ABCMeta.
    """
    def __init__(self, seed=None, name='synthetic', creator='user', remark='', 
                 **kwargs) -> None:
        self.seed = seed
        self.graph = None
        self.terminals = None
        self.name = name
        self.creator = creator
        self.remark = remark
        self.kwargs = kwargs
        
    def sample(self) -> nx.Graph:
        return self.graph
    
    def _generate_graph(self, *args, **kwrags) -> nx.Graph:
        pass
    
    def _add_graph_attribute(self) -> None:
        pass

    def _add_edge_attribute(self) -> None:
        pass
    

class STPInstance(AbstractInstance):
    def __init__(self, seed=None, name='synthetic', creator='user', remark='',  
                 max_edge_cost=MAX_EDGE_COST, **kwargs) -> None:
        super().__init__(seed, name, creator, remark, **kwargs)
        self.rng = kwargs.get('rng')
        if self.rng is None:
            self.rng = np.random.default_rng(self.seed)
        self.remark += re.sub(r'\s+', ' ', f"{self.seed}".replace('\n', ''))
        self.max_trials = 100
        self.max_edge_cost = max_edge_cost
        self.cost_type = self.kwargs.get('cost_type', 'gaussian')
        self.kwargs = kwargs
        
    def sample(self) -> nx.Graph:
        self.graph = self._generate_graph()
        self._sample_terminals()
        
        self._add_graph_attribute()
        self._add_edge_attribute()
        if self._check_feasibility():
            return self.graph
        else:
            raise RuntimeError("It is fail to generate graph instance.")
    
    # def _sample_terminals(self):
    #     num_terminals = self.rng.integers(low=2, high=int(0.2 * self.num_nodes))
    #     self.terminals = self.rng.choice(self.graph.nodes, num_terminals, replace=False)
        
    def _sample_terminals(self):
        is_terminal = self.rng.binomial(n=1, p=0.2, size=self.num_nodes)
        if np.sum(is_terminal) > 1:
            self.terminals = np.array([n for n, is_terminal in zip(self.graph.nodes, is_terminal) if is_terminal])
        else:
            self.terminals = self.rng.choice(self.graph.nodes, 2, replace=False)
        
    def _add_graph_attribute(self) -> None:
        graph_attr = {
            'Comment': {'Name': self.name, 
                        'Creator': self.creator, 
                        'Remark': ', '.join([self.remark, self.graph_model])},
            'Terminals': {'meta': {'numTerminals': len(self.terminals)}, 
                          'terminals': self.terminals},
            'Info': {'max_edge_cost': self.max_edge_cost},
            }
        self.graph.graph = graph_attr

    def _add_edge_attribute(self) -> None:
        if self.cost_type == 'gaussian':
            edge_cost = self._get_edge_cost_normal()
            
        elif self.cost_type == 'uniform' or self.graph_model == 'Grid':
            edge_cost = self._get_edge_cost_uniform()
            
        else:
            raise ValueError(f"Invalid cost type: {self.cost_type}")
            
        attrs = {edge: {'cost': cost} for edge, cost in zip(self.graph.edges(), edge_cost)}
        nx.set_edge_attributes(self.graph, attrs)
    
    def _get_edge_cost_uniform(self):
        edge_cost = self.rng.integers(low=1, high=self.max_edge_cost + 1, size=self.graph.number_of_edges())
        return edge_cost
        
    def _get_edge_cost_normal(self):
        mu = (1 + self.max_edge_cost) / 2
        sigma = (self.max_edge_cost - mu) / 3
        s = np.random.normal(mu, sigma, self.graph.number_of_edges())
        s = np.clip(s, 1, self.max_edge_cost)
        edge_cost = np.rint(s).astype(int)
        return edge_cost

    def __str__(self):
        return f"{self.graph_model}"
    
    def _check_feasibility(self):
        if self.graph is None:
            raise RuntimeError("Graph is not generated!")
        else:
            if not nx.is_connected(self.graph):
                raise RuntimeError("Graph is composed of multiple components.")
        return True
        

class STPInstance_test(STPInstance):
    def __init__(self, seed=None, name='synthetic', creator='user', remark='', **kwargs) -> None:
        super().__init__(seed=seed, name=name, creator=creator, remark=remark, **kwargs)
        
    def sample(self) -> nx.Graph:
        self.graph = self._generate_graph()
        self.terminals = [2, 8, 9]
        self._add_graph_attribute()
        self._add_edge_attribute()
        return self.graph
    
    def _generate_graph(self, *args, **kwrags) -> nx.Graph:
        graph = nx.erdos_renyi_graph(10, 0.5, seed=self.seed)
        return graph
    
    def _add_graph_attribute(self) -> None:
        graph_attr = {
            'Comment': {'Name': self.name, 
                        'Creator': self.creator, 
                        'Remark': self.remark},
            'Terminals': {'meta': {'numTerminals': len(self.terminals)}, 
                          'terminals': self.terminals}
            }
        self.graph.graph = graph_attr

    def _add_edge_attribute(self) -> None:
        np.random.seed(self.seed)
        for e in self.graph.edges:
            self.graph.edges[e]['cost'] = np.random.randint(low=1, high=10)

    
class STPInstance_erdos_renyi(STPInstance):
    def __init__(self, n: int, p: float = 1, seed=None, name='synthetic', creator='user', remark='', **kwargs) -> None:
        """Generate a random graph by Erdos-Renyi model G(n, p).
        https://en.wikipedia.org/wiki/Erdős–Rényi_model

        Args:
            n (int): number of nodes
            p (float): upper limit of probability of edge occurrence. Actual probability is determined by uniform sampling [p_lower, p] 
            seed (_type_, optional): _description_. Defaults to None.
            name (str, optional): _description_. Defaults to 'synthetic'.
            creator (str, optional): _description_. Defaults to 'user'.
            remark (str, optional): _description_. Defaults to ''.
        """
        super().__init__(seed, name, creator, remark, **kwargs)
        self.n = n
        self.p = p
        self.directed = kwargs.get('directed', False)
        self.graph_model = 'ER'
    
    def _generate_graph(self) -> nx.Graph:
        p_lower = np.log(self.n) / self.n # probability p such that G(n, p) almost surely be connected.
        p = self.rng.uniform(low=p_lower, high=self.p)
        graph = nx.erdos_renyi_graph(self.n, p, seed=self.rng, directed=self.directed)
        
        cnt = 0
        while not nx.is_connected(graph) and cnt < self.max_trials:
            graph = nx.erdos_renyi_graph(self.n, p, seed=self.rng, directed=self.directed)
            cnt += 1
        
        if not nx.is_connected(graph):
            raise Exception("Something wrong")
        
        self.num_nodes = graph.number_of_nodes()
        return graph
    
    def __repr__(self):
        return f"{self.graph_model}"
        
        

class STPInstance_watts_strogatz(STPInstance):
    def __init__(self, n: int, k: int=None, p: float=1, seed=None, name='synthetic', creator='user', remark='', **kwargs) -> None:
        """Generate a random graph by Watts–Strogatz model.
        https://en.wikipedia.org/wiki/Watts%E2%80%93Strogatz_model

        Args:
            n (int): number of nodes
            k (int, optional): Each node is joined with its k nearest neighbors in a ring topology. Defaults to None.
                                if it is 'None', 'k' is sampled within [3, 4, 5] with equal probability.
            p (float, optional): Maximum probability of rewiring each edge. Defaults to 1.
            seed (_type_, optional): _description_. Defaults to None.
            name (str, optional): _description_. Defaults to 'synthetic'.
            creator (str, optional): _description_. Defaults to 'user'.
            remark (str, optional): _description_. Defaults to ''.
        """
        super().__init__(seed, name, creator, remark, **kwargs)
        self.n = n
        self.k = k
        self.p = p
        self.graph_model = 'WS'
        
    def __repr__(self):
        return f"{self.graph_model}"
    
    def _generate_graph(self) -> nx.Graph:
        k = self.rng.integers(3, 6)
        p = self.rng.uniform(low=0, high=self.p)
        graph = nx.connected_watts_strogatz_graph(n=self.n, k=k, p=p, seed=self.rng)
        self.num_nodes = graph.number_of_nodes()
        return graph
        
    
class STPInstance_regular(STPInstance):
    def __init__(self, n: int, seed=None, name='synthetic', creator='user', remark='', **kwargs) -> None:
        """Generate a random graph by random regular model.
        https://en.wikipedia.org/wiki/Random_regular_graph

        Args:
            n (int): number of nodes
            seed (_type_, optional): _description_. Defaults to None.
            name (str, optional): _description_. Defaults to 'synthetic'.
            creator (str, optional): _description_. Defaults to 'user'.
            remark (str, optional): _description_. Defaults to ''.
        """
        super().__init__(seed, name, creator, remark, **kwargs)
        self.n = n
        self.graph_model = 'RR'
        
    def __repr__(self):
        return f"{self.graph_model}"
    
    def _generate_graph(self) -> nx.Graph:
        d = self.rng.integers(3, 5)
        graph = nx.random_regular_graph(d, self.n, seed=self.rng)
        
        cnt = 0
        while not nx.is_connected(graph) and cnt < 10:
            graph = nx.random_regular_graph(d, self.n, seed=self.rng)
            cnt += 1
        
        self.num_nodes = graph.number_of_nodes()
        return graph
        

class STPInstance_grid(STPInstance):
    def __init__(self, n: int, seed=None, name='synthetic', creator='user', remark='', **kwargs) -> None:
        """Generate a random (m x n x l) grid graph.
        
        Args:
            m (int): width
            n (int): height
            l (int): layer
            seed (_type_, optional): _description_. Defaults to None.
            name (str, optional): _description_. Defaults to 'synthetic'.
            creator (str, optional): _description_. Defaults to 'user'.
            remark (str, optional): _description_. Defaults to ''.
        """
        super().__init__(seed, name, creator, remark, **kwargs)
        self.graph_model = 'Grid'
        self.dim_candidate = [x for x in self.product_pairs(n) if min(x) >= 4]
        self.max_edge_cost = 2
        
    def __repr__(self):
        return f"{self.graph_model}"
    
    def _generate_graph(self) -> nx.Graph:
        dx, dy = self.rng.choice(self.dim_candidate, 1)[0]
        self.dx = dx
        self.dy = dy
        self.dz = 1
        
        self.num_nodes = self.dx * self.dy * self.dz
        adj_mat = self._build_grid(self.num_nodes, dim=(self.dz, self.dy, self.dx)) 
        graph = nx.from_scipy_sparse_array(adj_mat)
    
        graph.dx = self.dx
        graph.dy = self.dy
        graph.dz = self.dz
        return graph
        
    def _build_grid(self, num_nodes, dim):
        dx, dy, dz = dim # networkx dim. order (z, y, x)
        vertices_dim = np.arange(num_nodes).reshape([dx, dy, dz])
        edges_total = np.empty([0,2], dtype=np.int8)
        for i in range(3):
            nodes_left = np.delete(vertices_dim, -1, axis=i).reshape([-1,1])
            nodes_right = np.delete(vertices_dim, 0, axis=i).reshape([-1,1])
            edges = np.append(nodes_left, nodes_right, axis=1)
            edges_total = np.append(edges_total, edges, axis=0)
        row = edges_total[:, 0]
        col = edges_total[:, 1]
        data = np.ones(len(edges_total))
        return csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    
    def _add_graph_attribute(self):
        super()._add_graph_attribute()
        dim = {'x': self.dx, 'y': self.dy, 'z': self.dz}
        self.graph.graph['Info'].update({'dim': dim})
    
    @staticmethod
    def find_divisor_pairs(n):
        divisors = []
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                divisors.append(i)
                if i != n // i:  # Avoid duplicates if `i` is the square root of `n`
                    divisors.append(n // i)
        return divisors

    def product_pairs(self, n):
        pairs = []
        divisors = self.find_divisor_pairs(n)
        for divisor in sorted(divisors):
            counterpart = n // divisor
            pairs.append((divisor, counterpart))
        return pairs



if __name__ == "__main__":
    from matplotlib import pyplot as plt
    
    seed = 1234
    insts = [
        STPInstance_erdos_renyi(n=100, p=0.1, seed=seed),
        STPInstance_watts_strogatz(n=100, seed=seed),
        STPInstance_regular(n=100, seed=seed),
        STPInstance_grid(n=20, seed=seed)
        ]
    for inst in insts:
        g = inst.sample()
        # nx.draw(g)
        # plt.savefig(f'logs/sample_graph_{str(inst)}.png')
        print(inst)
        print(g.number_of_edges())
        print(g.number_of_nodes())
        print(g.graph)
