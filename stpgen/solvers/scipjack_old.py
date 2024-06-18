import multiprocessing
import os
import platform
import re
import subprocess
import time
import typing

# import cvxpy as cvx
import networkx as nx

from stpgen.datasets.io import SCIPParser, SteinLibFormatWriter


class SCIPJackSTP:
    """
    function 1. path -> read .stp file -> solve -> read results
    function 2. networkx.Graph -> write (temporal) .stp file -> read .stp file -> solve -> read results
    """
    def __init__(self, graph: nx.Graph=None, path: str=None, lock=None,
                 timelimit=None):
        os.makedirs('logs', exist_ok=True)
        self.graph = graph
        self.path = path
        if self.graph is None and self.path is None:
            raise ValueError("One of the graph and path should not be <None>!")
        
        if self.graph:
            self.tmp_problem_path = f"logs/tmp_prob_{id(self.graph)}.stp"
            
        self.limit_timeout = timelimit
        self.path_setting = 'stpsolver/settingsfile.set' # limits/time = 100
        self.debug_mode = False
        self.solution = None
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

    def solve(self, verbose=False):
        if self.debug_mode:
            path_log = 'logs/test1.log'
            if os.path.exists(path_log):
                os.remove(path_log)
        else:
            path_log = None
        
        instance_path = self._get_instance_path()
        self._change_timelimit_setting()
        cmd = f'{self.path_scipjack} -f {instance_path} -s {self.path_setting}' # -l {path_log}
        
        if path_log:
            cmd += f" -l {path_log}"
        if verbose:
            print(cmd)
            
        t0 = time.time()
        p1 = subprocess.run(args=[cmd], shell=True, capture_output=True)
        t1 = time.time()
        if p1.returncode == 0:
            p1_decode = p1.stdout.decode()
            if verbose:
                for l in p1_decode.split('\n'):
                    print(l)
            solution_parser = SCIPParser()
            output = solution_parser.parse(p1_decode)
            output.update(
                {
                    'instance_name': instance_path,
                    'time': t1 - t0,
                    'solving_method': 'scip-jack',
                }
            )
            # ouput solution as nx.Graph
            edges = output['solution']
            if self.graph:
                self.solution = self.graph.edge_subgraph(edges).copy()
                
                self.graph.graph['Terminals']
                self.graph.edges[(5, 12)]
                
            else:
                from stpgen.datasets.io import NetworkxMaker, SteinLibFormatReader
                self.path = self.tmp_problem_path
                instance = SteinLibFormatReader(offset=1).read(self.path)
                graph = NetworkxMaker().convert(instance)
                
                graph.nodes()
                graph.graph['Terminals']
                graph.edges[(5, 12)]
                
                self.graph = graph
                self.solution = graph.edge_subgraph(edges).copy()
        else:
            return None
        
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
        
    def _change_timelimit_setting(self):
        if self.limit_timeout:
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
            if eval(result) != self.limit_timeout:
                pattern = re.compile(r'(limits/time\s+=\s+).*?(\n)', re.DOTALL)
                def replace(match):
                    return match.group(1) + f"{self.limit_timeout}" + match.group(2)

                text_new = re.sub(pattern, replace, text)
                with open(self.path_setting, 'w') as file:
                    file.write(text_new)
                    

