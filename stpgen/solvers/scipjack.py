import multiprocessing
import os
import platform
import re
import subprocess
import time
import typing

# import cvxpy as cvx
import networkx as nx

from stpgen.datasets.io import (NetworkxMaker, SCIPParser,
                                 SteinLibFormatReader, SteinLibFormatWriter)


class SCIPJackRunner:
    """run SCIPJack command in terminal
    """
    def __init__(self, path_setting=None, timelimit=None) -> None:
        self.path_setting = 'stpsolver/settingsfile.set' if path_setting is None else path_setting
        self.timelimit = timelimit
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
    
    def run(self, instance_path, verbose=False, write_log: str = None):
        self._change_timelimit_setting()
        cmd = f'{self.path_scipjack} -f {instance_path} -s {self.path_setting} ' # -l {path_log}
        
        if write_log:
            cmd += f" -l {write_log}"
        
        if verbose:
            print(cmd)
        
        try:    
            t0 = time.time()
            proc = subprocess.run(args=[cmd], shell=True, capture_output=True)
            t1 = time.time()
            if proc.returncode != 0:
                raise RuntimeError(f"{cmd}")
        
        except Exception as e:
            print(e)
            raise e
        
        else:
            output = self.parse_output(instance_path, proc, t1-t0, verbose=verbose)
            return output
        
    def parse_output(self, instance_path, proc, t, verbose=False):
        proc_decode = proc.stdout.decode()
        if verbose:
            for l in proc_decode.split('\n'):
                print(l)
        solution_parser = SCIPParser()
        output = solution_parser.parse(proc_decode)
        output.update(
            {
                'instance_name': instance_path,
                'time': t,
                'solving_method': 'scip-jack',
            }
        )
        return output
    
    def _change_timelimit_setting(self):
        if self.timelimit:
            # identify
            pattern = re.compile(r'limits/time\s+=\s+(.*?)\n', re.DOTALL)
            with open(self.path_setting, 'r') as file:
                text = file.read()
                match = pattern.search(text)
                if match:
                    result = match.group(1)
                else:
                    raise ValueError(f"Unsupported patterns")
            # replace
            if eval(result) != self.timelimit:
                pattern = re.compile(r'(limits/time\s+=\s+).*?(\n)', re.DOTALL)
                def replace(match):
                    return match.group(1) + f"{self.timelimit}" + match.group(2)

                text_new = re.sub(pattern, replace, text)
                with open(self.path_setting, 'w') as file:
                    file.write(text_new)


class SCIPJackSTP:
    """
    function 1. path -> read .stp file -> solve -> read results
    function 2. networkx.Graph -> write (temporal) .stp file -> read .stp file -> solve -> read results
    """
    def __init__(self, graph: nx.Graph=None, path: str=None, lock=None,
                 timelimit=None, verbose=False, include_solution: bool = True):
        os.makedirs('logs', exist_ok=True)
        self.graph = graph
        self.path = path
        
        if self.graph is None and self.path is None:
            raise ValueError("One of the graph and path should be provided!")
        
        if self.graph:
            self.tmp_problem_path = f"logs/tmp_prob_{id(self.graph)}.stp"
        
        self.scipsolver = SCIPJackRunner(timelimit=timelimit)
        self.solution = None
        self.debug_mode = False
        self.verbose = verbose
        self.include_solution = include_solution
    
    def solve(self, verbose=False):
        instance_path = self._get_instance_path()
        output = self.scipsolver.run(instance_path, verbose=self.verbose)
        
        # match node index between SCIPjack solution and original problem
        inst_reader = SteinLibFormatReader(offset=1)
        instance = inst_reader.read(instance_path)
        graph = NetworkxMaker().convert(instance)
        self.graph = graph
        
        if self.include_solution:
            sol = output['solution']
            self.solution = graph.edge_subgraph(sol).copy()
        
            if not self._fast_feasilbity_check(self.graph, self.solution):
                raise RuntimeError("infeasible solution!")
            else:
                output.update({'solution': sol})
                self.solution.graph['solving_info'] = output
        else:
            self.solution = graph.edge_subgraph([]).copy()
            self.solution.graph['Info'].update({k: v for k, v in output.items() if k in ['cost', 'time']})
            
        if not self.debug_mode and hasattr(self, "tmp_problem_path"):
            os.remove(self.tmp_problem_path)
        
        return output
        
    def _get_instance_path(self):
        if self.path is not None:
            return self.path
        else:
            self._write_stp_file()
            return self.tmp_problem_path
    
    def _write_stp_file(self):
        writer = SteinLibFormatWriter(self.graph)
        writer.write()
        with open(self.tmp_problem_path, 'w') as file:
            file.write(writer.stp_string)
            
    def _index_matched_solution(self, node_order, solution):
        edgelist = []
        for n0, n1 in solution:
            edge = (node_order[n0], node_order[n1])
            edgelist.append(edge)
        return edgelist
            
    def _fast_feasilbity_check(self, problem, solution):
        terminals = problem.graph['Terminals']['terminals']
        return nx.is_connected(solution) and set(terminals).issubset(set(solution.nodes))
        

