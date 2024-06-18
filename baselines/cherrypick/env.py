import networkx as nx
import numpy as np
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, List
from baselines.common.utils import NodeFeature


def get_max_edge_cost_from_state(state):
    graph_info = state.graph.graph.get('Info') or state.graph.graph.get('info')
    max_cost = graph_info.get('max_edge_cost') if graph_info else None
    return max_cost if max_cost is not None else 1


@dataclass
class STPState:
    graph: nx.Graph = None
    partial_solution: List[Any] = field(default_factory=list)
    available_vertices: List[Any] = field(default_factory=list)
    distance: dict = field(default_factory=dict)
    cost: np.float32 = 0.0
    step: np.int32 = 0


class STPEnvironment:
    def __init__(self, instance_generator=None, seed=None, args=None) -> None:
        self.instance_generator = instance_generator
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.instance = None
        self.nterminals = None
        self.terminals = None
        self.reward_function = args.reward_function if args is not None else 'default'
        self.args = args
        
    def reset(self, instance=None):
        self._init_instance(instance)
        initial_vertex = self.rng.choice(self.terminals, 1)[0]
        self._transision(initial_vertex)
        self.cost_prev = 0
        
        state = self.state
        info = None
        return state, info
            
    def step(self, action):
        self._check_action_validity(action)
        self.cost_prev = self.state.cost
        self._transision(action)
        
        state = deepcopy(self.state)
        reward = self._get_reward()
        terminated = self._is_done()
        truncated = False
        info = None
        return state, reward, terminated, truncated, info
    
    def _init_instance(self, instance=None):
        if instance:
            self.instance = instance
        else:
            self.instance = self.instance_generator.sample()
        self.nterminals = self.instance.graph['Terminals']['meta']['numTerminals']
        self.terminals = self.instance.graph['Terminals']['terminals']
        self.state = STPState(graph=self.instance)
        self.max_edge_cost = get_max_edge_cost_from_state(self.state)
        
    def _transision(self, vertex):
        self.state.partial_solution.append(vertex)
        self.state.available_vertices = self._get_available_vertices()
        
        if vertex in self.terminals:
            if self.args.graph_type == 'Grid':
                self.state.distance = NodeFeature.cherrypick_agumented_coord(self.state.graph, self.state.partial_solution, 
                                                            K=2, normalize_edge_cost=True)
            else:
                self.state.distance = NodeFeature.cherrypick(self.state.graph, self.state.partial_solution, 
                                                            K=2, normalize_edge_cost=True)
        self.state.cost = self._get_current_cost()
        self.state.step += 1
        
    def _get_available_vertices(self):
        neighbors = set()
        for v in self.state.partial_solution:
            vs = [n for n in self.state.graph.neighbors(v) if n not in self.state.partial_solution]
            neighbors.update(vs)
        return list(neighbors)
    
    def _get_current_cost(self):
        graph = self.state.graph
        solution = self.state.partial_solution
        cost = 0 
        edges_sol = []
        if len(solution) > 1:
            for i, current_action in enumerate(solution[1:]):
                partial_solution_prev = solution[:(i+1)]
                edges = [((current_action, n), graph[current_action][n]) for n in partial_solution_prev if graph.has_edge(current_action, n)]
                edge_selected = sorted(edges, key=lambda x: x[1]['cost'])[0]
                edges_sol.append(edge_selected[0])
        
            for _, _, dt in graph.edge_subgraph(edges_sol).edges(data=True):
                cost += (dt['cost'] / self.max_edge_cost)
        return cost
    
    def _get_reward(self):
        if self.reward_function == 'default':
            return self._reward_function_default()
        elif self.reward_function == 'c0':
            return self._reward_function_c0()
        elif self.reward_function == 'cost':
            return self._reward_function_cost()
        else:
            raise NotImplementedError('Reward function not implemented!')
        
    def _reward_function_default(self):
        vertex = self.state.partial_solution[-1]
        if vertex in self.terminals:
            # Positive constant reward for adding a terminal
            # c = sum(self.instance[u][v]['cost'] / self.max_edge_cost for u, v in self.instance.edges(vertex)) / len(self.terminals)
            c = 10 / len(self.terminals)
            return - (self.state.cost - self.cost_prev) - sum(self.state.distance[vertex]) + c
        else:
            return - (self.state.cost - self.cost_prev) - sum(self.state.distance[vertex])
        
    def _reward_function_c0(self):
        vertex = self.state.partial_solution[-1]
        return - (self.state.cost - self.cost_prev) - sum(self.state.distance[vertex])
    
    def _reward_function_cost(self):
        return - (self.state.cost - self.cost_prev)
    
    def _check_action_validity(self, action):
        if action not in self.state.available_vertices:
            raise ValueError('Invalid action!')
        
    def _is_done(self):
        return set(self.terminals).issubset(self.state.partial_solution) or not self.state.available_vertices
        

if __name__ == "__main__":
    import sys
    sys.path.append('.')
    
    from stpgen.datasets.synthetic.instance import STPInstance_erdos_renyi

    # Make environment with a generated instance.
    instance_sampler = STPInstance_erdos_renyi(n=20, p=1, seed=1234)
    env = STPEnvironment(instance_sampler)
    print(env.instance.graph)
    
    # Perform a sanity check with a random policy
    state, info = env.reset()
    done = False
    while not done:
        action = env.rng.choice(state.available_vertices, 1)
        state, reward, terminated, truncated, info, done = env.step(action)
        print(f"Observation:  Action: {action}, Reward: {reward}, Done: {done}, step: {state.step}, cost: {state.cost}, cost: {env.cost_prev}")
    print('okay')
    
    # This code demonstrates basic functionality of the STP environment with a random policy.

    # from baselines.cherrypick.agents import DQNAgent
    # from baselines.cherrypick.env import STPEnvironment
    # from baselines.cherrypick.networks import Vulcan
    # from stpgen.datasets.synthetic.instance import STPInstance_erdos_renyi

    # # Make environment with a generated instance.
    # instance_sampler = STPInstance_erdos_renyi(n=20, p=1, seed=1234)
    # env = STPEnvironment(instance_sampler)

    # args = tyro.cli(Args)
    # agent = DQNAgent(network=Vulcan, args=args)

    # # Perform a sanity check with a random policy
    # for _ in range(3):
    #     state, info = env.reset()
    #     print(env.instance.graph)

    #     terminated = False
    #     while not terminated:
    #         # action = env.rng.choice(state.available_vertices, 1)[0]
    #         action = agent.get_action(state)
            
    #         state, reward, terminated, truncated, info = env.step(action)
    #         msg = ''
    #         msg += f"Observation:  Action: {action}, Reward: {reward}, "
    #         msg += f"Done: {terminated}, step: {state.step}, "
    #         msg += f