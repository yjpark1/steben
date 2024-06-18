
import abc
import networkx as nx
import os
from collections import defaultdict, namedtuple
import re
import itertools
import numpy as np
import typing
import pathlib
from stpgen.datasets.synthetic.instance import MAX_EDGE_COST


data_root = 'data/'


class InstanceReader(metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        pass
    
        
    
class Extractor:
    def __init__(self) -> None:
        pass
    
    def extract(self, path_src, path_dst, exist_ok=True):
        """
        examples)
        path_src = 'data/SteinLib/raws/LIN.tgz'
        path_dst = 'data/SteinLib/extracted/LIN/'
        """
        path_src_ = pathlib.Path(path_src)
        extn = path_src_.suffix
        if extn == '.tgz':
            pass
        else:
            path_dst += path_src_.stem
        
        try:
            from pyunpack import Archive
            os.makedirs(path_dst, exist_ok=exist_ok)
            path_dst_ = '/'.join(path_dst.split('/')[:-1])
            Archive(path_src).extractall(path_dst_, auto_create_dir=True)
        except Exception as e:
            # raise e
            print(e, path_src)



class NetworkxMaker:
    """Make nx.Graph from a graph dict. 
    """
    def __init__(self) -> None:
        pass
    
    def convert(self, instance: typing.Dict) -> nx.Graph:
        if type(instance) is not dict:
            raise ValueError(f"Unavailable type for instance: {type(instance)}")
        g = nx.from_edgelist(instance['Graph']['edgelist'])
        
        if instance['Graph']['meta']['is_directed']:
            g = g.to_directed()
        
        attrs_graph = ['header', 'Comment', 'Terminals']
        attrs_graph = {k: instance[k] for k in attrs_graph}
        
        info = attrs_graph['Comment'].get('Info')
        if info:
            attrs_graph.update({'Info': eval(info)})
            attrs_graph['Comment'].pop('Info')
        
        g.graph.update(attrs_graph)
        
        coordinates = instance.get('Coordinates', False)
        if coordinates:
            for node, coordinate in coordinates.items():
                g.nodes[node]['coordinate'] = coordinate
        return self.reorder_nodes(g)
    
    @staticmethod
    def reorder_nodes(instance):
        inst = nx.Graph()
        inst.add_nodes_from(sorted(instance.nodes(data=True)))
        inst.add_edges_from(instance.edges(data=True))
        inst.graph = instance.graph
        return inst
        
        
class SteinLibFormatReader(InstanceReader):
    """Read STP file, and process to dict format.

    Args:
        InstanceReader (_type_): _description_
    
    .stp file format: https://steinlib.zib.de/format.php
    """
    def __init__(self, offset=0) -> None:
        super().__init__()
        self.instance = defaultdict(list)
        self.instance['header'] = None
        self.offset = offset
        self.node_order = []
        
    def read(self, path):
        stplines = self._read_file(path)
        self._process_to_dict(stplines)
        instance = self._process_sections()
        self._get_node_order(instance['Graph']['edgelist'])
        return instance
    
    def unique_preserve_order(self, arr):
        seen = set()
        unique = []
        for x in arr:
            if x not in seen:
                unique.append(x)
                seen.add(x)
        return np.array(unique)
        
    def _get_node_order(self, edge_list):
        nodelist = np.array([(n0, n1) for n0, n1, _ in edge_list])
        nodelist = np.concatenate((nodelist[:, 0], nodelist[:, 1]))
        self.node_order = self.unique_preserve_order(nodelist)
        
    def _read_file(self, path):
        try:
            with open(path, 'r') as file:
                stplines = file.readlines()
        except Exception as e:
            raise e
        else:
            return stplines
        
    def _process_to_dict(self, stplines):
        section = None
        for i, line in enumerate(stplines):
            if i == 0:
                self.instance['header'] = line.strip()
            
            if 'SECTION' in line:
                section = line.strip().split(' ')[1]
            elif 'END' in line:
                section = None
            else:        
                if section is not None:
                    self.instance[section].append(line.strip())
    
    def _process_sections(self):
        instance_processed = {}
        parser = STPSectionParser(offset=self.offset)
        for section_name, section in self.instance.items():
            if section_name == 'Comment':
                items = parser.parse_comment(section)
            elif section_name == 'Graph':
                items = parser.parse_graph(section)
            elif section_name == 'Terminals':
                items = parser.parse_terminals(section)
            elif section_name == 'MaximumDegrees':
                items = parser.parse_maxdegree(section)
            elif section_name == 'Coordinates':
                items = parser.parse_coordinates(section)
            elif section_name == 'header':
                items = section
            else:
                raise NotImplementedError(f"Unsupported keys [{section_name}] in .stp file")
            instance_processed[section_name] = items
        return instance_processed
        

class STPSectionParser:
    def __init__(self, offset=0) -> None:
        self.offset = offset
        self.Edge = namedtuple('Edge', ['src', 'dst', 'cost'])
        # self.Coordindates = namedtuple('Coords', ['node', 'x', 'y'])
    
    def parse_comment(self, section):
        section_processed = {}
        for key in ['Name', 'Date', 'Creator', 'Remark', 'Info', 'Problem']:
            regex = rf"^{key}(?:\s+)\"(.+)\"$"
            values = list(itertools.chain(*map(lambda x: re.findall(regex, x), section)))
            for value in values:
                section_processed[key] = value
        return section_processed
    
    def parse_graph(self, section):
        section_processed = {
            'meta': {},
            'edgelist': []
        }
        
        for line in section:
            line_splitted = line.split(' ')
            header = line_splitted[0]
            if header in ['Obstacles', 'Nodes', 'Edges', 'Arcs']:
                if header == 'Arcs':
                    header = 'Edges'
                    section_processed['meta']['is_directed'] = True
                elif header == 'Edges':
                    section_processed['meta']['is_directed'] = False
                else:
                    pass
                section_processed['meta'][f'num{header}'] = eval(line_splitted[1])
                
            elif header in ['E', 'A']:
                edge = self._parse_single_edge(edgeline=line_splitted[1:])
                section_processed['edgelist'].append(edge)
            else:
                raise TypeError(f"Unsupported edge type")
        self.check_num_edges(section_processed)
        return section_processed
        
    def _parse_single_edge(self, edgeline):
        try:
            edge = self.Edge(*[eval(x) for x in edgeline])
            edge = (edge.src - self.offset, edge.dst - self.offset, {'cost': edge.cost})
        except Exception as e:
            raise e
        else:
            return edge
        
    def parse_terminals(self, section):
        section_processed = {
            'meta': {},
            'terminals': []
        }
        for line in section:
            line_splitted = line.split(' ')
            header = line_splitted[0]
            if header in ['Terminals']:
                section_processed['meta'][f'num{header}'] = int(line_splitted[1])
                assert len(line_splitted) == 2
            elif header == 'T':
                section_processed['terminals'].append(int(line_splitted[1]) - self.offset)
                assert len(line_splitted) == 2
            else:
                raise NotImplementedError()
        
        self.check_num_terminals(section_processed)
        return section_processed
    
    def parse_maxdegree(self, section):
        raise NotImplementedError()
    
    def parse_coordinates(self, section):
        section_processed = {}
        for line in section:
            line_splitted = line.split(' ')
            line_splitted = [eval(x) for x in line_splitted[1:]]
            section_processed[line_splitted[0]] = np.array(line_splitted[1:])
        return section_processed
    
    def check_num_edges(self, section):
        assert section['meta']['numEdges'] == len(section['edgelist'])
    
    def check_num_terminals(self, section):
        assert section['meta']['numTerminals'] == len(section['terminals'])
        
        
class SteinLibFormatWriter:
    """Convert from networkx.Graph into .stp file.
    """
    def __init__(self, graph:nx.Graph) -> None:
        self.graph = graph
        self.file_header = '33D32945 STP File, STP Format Version 1.0'
        self.stp_string = ""
        self._is_index_started_from_zero()
        
    def _is_index_started_from_zero(self):
        if min(self.graph.nodes()) == 0:
            self.offset = 1
        else:
            self.offset = 0
    
    def write(self, filepath=None) -> str:
        self._write_header()
        self._write_comment()
        self._write_graph()
        self._write_terminals()
        self._write_EOF()
        
        if filepath is None:
            filepath = 'logs/test.stp'
            
        with open(filepath, 'w') as file:
            file.write(self.stp_string)
        
    def _write_header(self):
        header = self.graph.graph.get('header', self.file_header)
        self.stp_string += header
        self.stp_string += '\n\n'
        
    def _write_comment(self):
        empty_word = None
        comment = "SECTION Comment\n"
        for k in ['Name', 'Creator', 'Remark']:
            v = self.graph.graph['Comment'].get(k, empty_word)
            if v:
                space = ' ' * (8 - len(k))
                comment += f"{k}{space}\"{v}\"\n"
                
        info = self.graph.graph.get('Info')
        if info:
            k = 'Info'
            space = ' ' * (8 - len(k))
            comment += f"{k}{space}\"{info}\"\n"
                
        comment += 'END\n\n'
        self.stp_string += comment
        
    def _write_graph(self):
        graph_lines = "SECTION Graph\n"
        graph_lines += f"Nodes {self.graph.number_of_nodes()}\n"
        graph_lines += f"Edges {self.graph.number_of_edges()}\n"
        
        edge_header = "E" if not self.graph.is_directed() else "A"
        graph_lines += '\n'.join([f"{edge_header} {i + self.offset} {j + self.offset} {d['cost']}" for i, j, d in self.graph.edges(data=True)])
        graph_lines += '\nEND\n\n'
        self.stp_string += graph_lines
        
    def _write_terminals(self):
        terminal_lines = "SECTION Terminals\n"
        terminal_lines += f"Terminals {self.graph.graph['Terminals']['meta'].get('numTerminals')}\n"
        terminal_lines += '\n'.join([f"T {i + self.offset}" for i in self.graph.graph['Terminals']['terminals']])
        terminal_lines += '\nEND\n\n'
        self.stp_string += terminal_lines

    def _write_EOF(self):
        self.stp_string += "EOF\n"
        
        
class SCIPParser:
    """Parse SCIP log for output
    """
    def __init__(self, offset_node_index=0):
        self.output = {}
        self.offset_node_index = offset_node_index
    
    def parse(self, log):
        """get string from the SCIP log
        
        Args:
            log (_type_): _description_
        """
        # log_split = log.split('\n')
        # [x for x in log_split if 'solve problem' in x]
        # log_split[0].startswith('')
        
        # log_path = '/Users/yj.park/Documents/PythonProjects/scipstp/logs/test1.log'
        # with open(log_path, 'r') as f:
        #     log = f.read()

        patterns = {
            'status': r'SCIP Status\s+:\s+(.*?)\n',
            'solution': r'primal solution \(original space\):\n(.*?)(?=Statistics\n)',
            'objective value': r'objective value:\s+(.*?)\n',
            'gap': r'Gap\s+:\s+(.*?%)\n',
        }
        
        log_parsed = {}
        for key, regex in patterns.items():
            value = self._get_string_from_regex(regex, log)
            log_parsed[key] = value
        self._make_output(log_parsed)
        return self.output

    def _get_string_from_regex(self, regex, text):
        pattern = re.compile(regex, re.DOTALL)

        match = pattern.search(text)

        if match:
            result = match.group(1)
            return result
        else:
            # raise Exception("No pattern founded.")
            print("No pattern founded.")
            return None
            
    def _make_output(self, log_parsed):
        obj = log_parsed['objective value']
        if obj is not None:
            self.output['cost'] = int(obj)
        self.output['solution'] = None
        
        sol = log_parsed['solution']
        sol = [x for x in sol.split('\n') if 'x_' in x]
        if len(sol):
            edgelist = list(map(self._get_edge_from_a_line, sol))
            self.output['solution'] = edgelist
        
    def _get_edge_from_a_line(self, line):
        pattern = re.compile(r'x_(\d+)_(\d+)')
        match = pattern.search(line)

        if match:
            value1 = int(match.group(1)) + self.offset_node_index
            value2 = int(match.group(2)) + self.offset_node_index
            return (value1, value2)
        return None
            
        
    
if __name__ == "__main__":
    # sample code
    path_src = 'data/SteinLib/raws/LIN.tgz'
    path_dst = 'data/SteinLib/extracted/'
    Extractor().extract(path_src, path_dst)
    
    a = SteinLibFormatReader()
    path = 'data/SteinLib/extracted/LIN/lin02.stp'
    instance = a.read(path)
    
    b = NetworkxMaker()
    g = b.convert(instance)
    