import abc
from typing import Literal, get_args
from stpgen.datasets.io import SteinLibFormatReader, NetworkxMaker
import subprocess
import time
import os
import re
import platform
import networkx as nx
import math
import abc


class GenericSolver(metaclass=abc.ABCMeta):
    """_summary_

    Args:
        metaclass (_type_, optional): _description_. Defaults to abc.ABCMeta.
    """
    def __init__(self) -> None:
        self.__methods_supported = ['scip-jack', 'scip', '2-approximation_nx', '2-approximation_sp']
        self.msg_error_method = lambda method: ValueError(f"unsupported method for [{method}]")
        self.method = None
        self.path_scip = None
        self.verbose = True
        os.makedirs('logs', exist_ok=True)
        self._set_scipjack_path()
        
    def _set_scipjack_path(self):
        os_ = platform.platform().lower()
        if 'mac' in os_:
            path_scipjack = 'stpsolver/build_mac_arm/scip/bin/applications/scipstp'
        elif 'linux' in os_:
            path_scipjack = 'stpsolver/build_linux/scip/bin/applications/scipstp'
        else:
            raise ValueError(f"Unsupported OS {os_}")
        self.path_scipjack = path_scipjack
        
    def solve_instance(self, instance, method: Literal['scip-jack','scip', '2-approximation_nx', '2-approximation_sp']):
        raise NotImplementedError
        


class STPfileSolver(GenericSolver):
    def __init__(self) -> None:
        super().__init__()
        self.instance_path = None
    
    def solve_instance(self, path: str, method: Literal['scip-jack','scip', '2-approximation_nx', '2-approximation_sp']):
        self.__init__()
        if method in self.__methods_supported:
            self.method = method
        else: 
            raise self.msg_error_method(method)
        self.instance_path = path
        print(self.instance_path)
        
        if method == 'scip-jack':
            output = self._solve_scipjack(path)
        elif method == 'scip':
            output = self._solve_scip(path)
        elif method == '2-approximation_nx':
            output = self._solve_2approximation(path)
        elif method == '2-approximation_sp':
            output = self._solve_2approximation_scipy(path)
        else:
            raise self.msg_error_method(method)
        return output
        
    def _read_instance_and_convert_to_networkx(self, path):
        instance = SteinLibFormatReader().read(path)
        graph = NetworkxMaker().convert(instance)
        graph.name = path
        return graph
    
    def _solve_scipjack(self, instance_path):
        # path_log = 'logs/test1.log'
        # if os.path.exists(path_log):
        #     os.remove(path_log)
        
        path_setting = 'settingsfile.set'
        # limits/time = 100
        cmd = f'{self.path_scipjack} -f {instance_path} -s {path_setting}' # -l {path_log}
        # process = subprocess.Popen(cmd, shell=True)
        print(cmd)
        t0 = time.time()
        p1 = subprocess.run(args=[cmd], shell=True, capture_output=True)
        t1 = time.time()
        if p1.returncode == 0:
            p1_decode = p1.stdout.decode()
            if self.verbose:
                for l in p1_decode.split('\n'):
                    print(l)
            from stpgen.datasets.io import SCIPParser
            solution_parser = SCIPParser()
            output = solution_parser.parse(p1_decode)
            output.update(
                {'instance_name': self.instance_path,
                'time': t1 - t0,
                'solving_method': self.method,}
            )
        else:
            return None
        return output
    
    def _solve_scip(self, instance_path):
        from stpgen.solvers.cvxpystp import CVXSTP, CVXSTPvec
        graph = self._read_instance_and_convert_to_networkx(instance_path)
        
        solver = CVXSTPvec(graph)
        solver.solve(self.verbose)
        if solver.solution is not None:
            cost_for_solution = sum([graph.get_edge_data(s, t)['cost'] for s, t in solver.solution.edges()])
            if not math.isclose(solver.problem.value, cost_for_solution):
                raise ValueError(f"different objective value on {self.instance_path}")
            output = {
                'instance_name': self.instance_path,
                'cost': cost_for_solution,
                'solution': list(solver.solution.edges),
                'time': solver.problem._solve_time,
                'solving_method': self.method,
            }
            return output
        else:
            return None
    
    def _solve_2approximation(self, instance_path):
        graph = self._read_instance_and_convert_to_networkx(instance_path)
        t0 = time.time()
        st = nx.approximation.steiner_tree(graph, sorted(graph.graph['Terminals']['terminals']), weight='cost', method='kou')
        t1 = time.time()
        cost_for_solution = sum([dt['cost'] for src, dst, dt in st.edges(data=True)])
        output = {
            'instance_name': self.instance_path,
            'cost': cost_for_solution,
            'solution': list(st.edges),
            'time': t1 - t0,
            'solving_method': self.method,
        }
        return output
        
    def _solve_2approximation_scipy(self, instance_path):
        from stpgen.solvers.heuristic import TwoApprox
        
        graph = self._read_instance_and_convert_to_networkx(instance_path)
        t0 = time.time()
        st = TwoApprox(graph, sorted(graph.graph['Terminals']['terminals']), weight='cost')
        t1 = time.time()
        cost_for_solution = sum([graph.get_edge_data(s, t)['cost'] for s, t in st.edges()])
        output = {
            'instance_name': self.instance_path,
            'cost': cost_for_solution,
            'solution': list(st.edges),
            'time': t1 - t0,
            'solving_method': self.method,
        }
        return output
        
        
class NetworkxSolver(GenericSolver):
    def __init__(self) -> None:
        super().__init__()
    
    def solve_instance(self, path: str, method):
        pass
    


