"""
https://www.philipzucker.com/cvxpy-and-networkx-flow-problems/
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4927437/pdf/nihms772320.pdf
"""
import sys, os
import networkx as nx
import cvxpy as cvx
import matplotlib.pyplot as plt
import numpy as np
import math
import warnings
import multiprocessing
import time
import signal
import pickle



# SCIPOPTDIR = stpsolver/build_linux/scip

def timeout_handler(num, stack):
    raise TimeoutError('Timeout')


class CVXSTP:
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.problem = None
        self.objective = None
        self.constraints = None
        self.solution = None

    def solve(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            self._model()
            self.problem = cvx.Problem(self.objective, self.constraints)
            try:
                # self.problem.solve(solver=cvx.SCIP, scip_params={
                #     "limits/time": 1}, verbose=True)
                
                proc = multiprocessing.Process(target=self.problem.solve, 
                                               kargs={'solver': 'SCIP', 
                                                      'verbose': False,
                                                      'scip_params': {"limits/time": 1}})
                proc.start()
                proc.join(1)
                if proc.is_alive():
                    proc.terminate()
                    proc.join()
                    raise TimeoutError("time out!")
                
            except cvx.SolverError as e:
                print(e, self.graph.name)
            else:
                self.solution = self._get_solution()
        
    def _model(self):
        graph_directed = self.graph.to_directed()
        # set terminals
        terminals = self.graph.graph['Terminals']['terminals']
        Nk = len(terminals) - 1

        for e in graph_directed.edges():
            graph_directed.edges[e]['cvx_var_flow'] = cvx.Variable(shape=(Nk,))

        obj = 0
        for i, j, dt in self.graph.edges(data=True):
            self.graph.edges[(i, j)]['cvx_var_decision'] = cvx.Variable(shape=(1,), integer=True)
            obj += dt['cost'] * self.graph.edges[(i, j)]['cvx_var_decision']

        # objective
        self.objective = cvx.Minimize(obj)
        
        # constraints
        constraints = []
        for k, t in enumerate(terminals[1:]):
            for node in self.graph.nodes():
                inflow = 0
                for i, j, dt in graph_directed.in_edges(node, data=True): # edges to node -> for inflow
                    inflow += dt['cvx_var_flow'][k]
                outflow = 0    
                for i, j, dt in graph_directed.out_edges(node, data=True): # edges from node -> for outflow
                    outflow += dt['cvx_var_flow'][k]

                netflow = inflow - outflow
                
                if node == terminals[0]:
                    value = -1
                elif node == t:
                    value = 1
                else:
                    value = 0
                
                constraints.append(netflow == value)
            
        for i, j, dt in graph_directed.edges(data=True):
            constraints.append(dt['cvx_var_flow'] <= self.graph.edges[(i, j)]['cvx_var_decision'])
            constraints.append(dt['cvx_var_flow'] >= 0)

        for i,j, dt in self.graph.edges(data=True):
            constraints.append(dt['cvx_var_decision'] >= 0)
            constraints.append(dt['cvx_var_decision'] <= 1)
        
        self.constraints = constraints
        
    def _get_solution(self):
        edges = []
        for i, j, dt in self.graph.edges(data=True):
            if math.isclose(dt['cvx_var_decision'].value, 1):
                edges.append((i, j))
        subgraph = self.graph.edge_subgraph(edges)
        return subgraph
    

class CVXSTPvec:
    def __init__(self, graph: nx.Graph, timelimit=1e+20):
        self.graph = graph
        self.problem = None
        self.objective = None
        self.constraints = None
        self.solution = None
        self.limit_timeout = int(timelimit)
        self.debug_mode = False

    def solve(self, verbose=False):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            self._model()
            self.problem = cvx.Problem(self.objective, self.constraints)
            
            signal.signal(signal.SIGALRM, timeout_handler)
            if not self.debug_mode and False:
                timelimit = min(sys.maxsize, self.limit_timeout + 10)
                signal.alarm(timelimit) # compile time
            
            try:
                self.problem.solve(solver=cvx.SCIP, scip_params={
                    "limits/time": self.limit_timeout}, verbose=verbose)
                
            except cvx.SolverError as e:
                print(e, self.graph.name)
            except TimeoutError as e:
                print(e, self.graph.name)
            else:
                self.solution = self._get_solution()
            finally:
                signal.alarm(0)
        
    def _model(self):
        if self.graph.is_directed():
            self._model_directed()
        else:
            self._model_undirected()
        
    def _model_directed(self):
        # set terminals
        terminals = self.graph.graph['Terminals']['terminals']
        Nk = len(terminals) - 1

        num_edges = self.graph.number_of_edges()
        num_nodes = self.graph.number_of_nodes()
        
        var_flow = cvx.Variable(shape=(num_edges, Nk), name='flow')
        self.var_decision = cvx.Variable(shape=num_edges, boolean=True, name='decision')
        
        edgecost = cvx.Constant(np.array([d['cost'] for _, _, d in self.graph.edges(data=True)]))
        self.objective = cvx.Minimize(cvx.sum(cvx.multiply(edgecost, self.var_decision)))
        
        constraints = []
        constraints.append(var_flow >= 0)
        
        Lcvx = cvx.Constant(nx.incidence_matrix(self.graph, oriented=True))
        node_index = dict(zip(self.graph.nodes(), range(num_nodes)))
        
        for k, t in enumerate(terminals[1:]):
            source = np.zeros(num_nodes)
            source[node_index[terminals[0]]] = -1 # positive flow only
            source[node_index[t]] = 1 # negative flow only
            flow = var_flow[:, k]
            constraints.append(Lcvx @ flow == source) # (-1 * sum of positive flow) + (sum of negative flow)
            constraints.append(var_flow[:, k] <= self.var_decision)
        self.constraints = constraints
        
    def _model_undirected(self):
        # set terminals
        terminals = self.graph.graph['Terminals']['terminals']
        Nk = len(terminals) - 1

        num_edges = self.graph.number_of_edges()
        num_nodes = self.graph.number_of_nodes()
        var_posflow = cvx.Variable(shape=(num_edges, Nk), name='posflow')
        var_negflow = cvx.Variable(shape=(num_edges, Nk), name='negflow')
        # positive flow of edge (i, j) := flow from node i to node j (= f_ij)
        # negative flow of edge (i, j) := flow from node j to node i (= f_ji)
        
        self.var_decision = cvx.Variable(shape=num_edges, boolean=True, name='decision')
        
        edgecost = cvx.Constant(np.array([d['cost'] for _, _, d in self.graph.edges(data=True)]))
        self.objective = cvx.Minimize(cvx.sum(cvx.multiply(edgecost, self.var_decision)))
        
        constraints = []
        constraints.append(var_posflow >= 0)
        constraints.append(var_negflow >= 0)
        
        Lcvx = cvx.Constant(nx.incidence_matrix(self.graph, oriented=True))
        node_index = dict(zip(self.graph.nodes(), range(num_nodes)))
        
        for k, t in enumerate(terminals[1:]):
            source = np.zeros(num_nodes)
            source[node_index[terminals[0]]] = -1 # positive flow only
            source[node_index[t]] = 1 # negative flow only
            flow = var_posflow[:, k] - var_negflow[:, k]
            constraints.append(Lcvx @ flow == source) # (-1 * sum of positive flow) + (sum of negative flow)
        
            constraints.append(var_negflow[:, k] <= self.var_decision)
            constraints.append(var_posflow[:, k] <= self.var_decision)
        self.constraints = constraints
        
    def _get_solution(self):
        edges = []
        for x, edge in zip(self.var_decision.value, 
                            self.graph.edges()):
            if math.isclose(x, 1):
                edges.append(edge)
        subgraph = self.graph.edge_subgraph(edges).copy()
        return subgraph