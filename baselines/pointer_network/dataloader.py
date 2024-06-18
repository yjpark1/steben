import pickle
import glob
import re
from torch.utils.data import Dataset
import sys, os
import torch
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from common.utils import NodeFeature
import numpy as np
from itertools import zip_longest
import networkx as nx
from tqdm import tqdm
from collections import deque

class DataWrapper:
    def __init__(self, Xb, Yb, length, info, As, Cs, opt, Rs, Ts):
        self.Xb = Xb
        self.Yb = Yb
        self.length = length
        self.info = info
        self.As = As
        self.Cs = Cs
        self.opt = opt
        self.Rs = Rs
        self.Ts = Ts
        
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return DataWrapper(
                self.Xb[idx],
                self.Yb[idx],
                self.length[idx],
                self.info[idx],
                self.As[idx],
                self.Cs[idx],
                self.opt[idx],
                self.Rs[idx],
                self.Ts[idx],
            )
        elif isinstance(idx, int):
            return (
                self.Xb[idx],
                self.Yb[idx],
                self.length[idx],
                self.info[idx],
                self.As[idx],
                self.Cs[idx],
                self.opt[idx],
                self.Rs[idx],
                self.Ts[idx],
            )
        else:
            raise TypeError("Invalid argument type.")
    
    def __len__(self):
        return len(self.Xb)
    
    def __repr__(self):
        return (f"DataWrapper(\n"
                f"  Xb={self.Xb},\n"
                f"  Yb={self.Yb},\n"
                f"  length={self.length},\n"
                f"  info={self.info},\n"
                f"  As={self.As},\n"
                f"  Cs={self.Cs},\n"
                f"  opt={self.opt}\n"
                f"  Rs={self.Rs},\n"
                f"  Ts={self.Ts},\n"
                f")")
        
    def concatenate(self, other):
        if not isinstance(other, DataWrapper):
            raise TypeError("The argument must be an instance of DataWrapper")
        return DataWrapper(
            self.Xb + other.Xb,
            self.Yb + other.Yb,
            self.length + other.length,
            self.info + other.info,
            self.As + other.As,
            self.Cs + other.Cs,
            self.opt + other.opt,
            self.Rs + other.Rs,
            self.Ts + other.Ts,
        )
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.children = []


class SolutionGenerator():
    def __init__(self, no_label=False):
        self.no_label = no_label
        self.root = -1
        
    def run(self, solution, terminals):
        self._init_root(terminals)
        if self.no_label: return []
        return self.generate_node_list(solution, terminals)
    
    def _init_root(self, terminals):
        self.root = max(terminals)
        
    def graph_to_tree(self, G, root):
        tree_node = TreeNode(root)
        visited = set()
        queue = [(tree_node, root)]
        while queue:
            parent_node, parent_value = queue.pop(0)
            visited.add(parent_value)
            nodes = sorted(G.neighbors(parent_value))[::-1]
            for neighbor in nodes:
                if neighbor not in visited:
                    child_node = TreeNode(neighbor)
                    parent_node.children.append(child_node)
                    queue.append((child_node, neighbor))
        return tree_node

    def generate_node_list(self, solution, terminals):
        tree = self.graph_to_tree(solution, self.root)
        output = []
        self.nary_level_order(tree, output)
        return output

    def nary_level_order(self, root, output):
        if root is None:
            return
        queue = deque([root])
        while queue:
            node = queue.popleft()
            # print(node.value, end=' ')
            output.append(node.value)
            for child in node.children:
                queue.append(child)

    def generate_edge_tuple(self, solution):
        """
        change solution format

        Args:
            solution (nx.Graph): _description_

        Returns:
            list: _description_
            
        References: (Vinyals, O., Fortunato, M., & Jaitly, N. (2015). Pointer networks. Advances in neural information processing systems, 28.)
        ...
        we order the triangles Ci by their incenter coordinates (lexicographic order) and choose the increasing triangle representation2. 
        Without ordering, the models learned were not as good, and finding a better ordering that the Ptr-Net could better exploit is part of future work.
        ...
        
        """
        return sorted([(n1, n2) if n1 < n2 else (n2, n1) for n1, n2 in solution.edges])



class SplitDataset():
    def __init__(self, total_data_size, batch_size, g='er', size=20, phase='train', no_label=False):
        self.total_data_size = total_data_size
        self.batch_size = batch_size
        self.g=g
        self.size=size
        self.phase=phase
        self.no_label = no_label
        self.raw_paths = self.read_raw_paths(g, size, phase) if not phase.startswith('valid') else self.read_raw_paths(g, size, phase.replace('valid', 'train'))[-2:-1]
        self.sol_generator = SolutionGenerator(no_label)
        self.RAW_N_SAMPLES_PER_PKL = 100 if self.size == 1000 else 10000
        assert self.total_data_size <= len(self.raw_paths) * self.RAW_N_SAMPLES_PER_PKL  # 1 pkl = 10000 instances
        self.chunk_size = (total_data_size + batch_size -1) // batch_size    # ceil
        
    def save(self, output_dir):  # split raw into chunk
        """
        process features and save chunked dataset
        Args:
            output_dir (str)
        Returns:
            str: index_file path
        """
    
        to_be_saved = False     # check files
        for chunk_file_idx in range(self.chunk_size):
            chunk_file = os.path.join(output_dir, f'chunk_{chunk_file_idx}.pkl')
            if not os.path.exists(chunk_file):
                print(chunk_file,' is missing')
                to_be_saved = True
                break

        if to_be_saved:
            assert self.RAW_N_SAMPLES_PER_PKL >= self.batch_size, "if three files are needed for a single batch, it raises error"
            raw_file_idx, raw_within_idx  = -1, -1
            data, n_data = None, -1
            
            for chunk_file_idx in tqdm(range(self.chunk_size)):
                chunk_file = os.path.join(output_dir, f'chunk_{chunk_file_idx}.pkl')
                if os.path.exists(chunk_file): continue
                batch_size = min(self.batch_size, self.total_data_size - (chunk_file_idx*self.batch_size))  # current batch_size
                if raw_file_idx != ((chunk_file_idx * self.batch_size) // self.RAW_N_SAMPLES_PER_PKL):  # if not loaded before
                    raw_file_idx = (chunk_file_idx * self.batch_size) // self.RAW_N_SAMPLES_PER_PKL
                    data, n_data = self.read_pkl(self.raw_paths[raw_file_idx])
                raw_within_idx = (chunk_file_idx * self.batch_size) % self.RAW_N_SAMPLES_PER_PKL
                
                if n_data >= raw_within_idx + batch_size:  # loaded data cover batch
                    chunk_data = data[raw_within_idx:raw_within_idx+batch_size]
                    raw_within_idx += batch_size
                else:   # can't cover batch
                    chunk_data = data[raw_within_idx:]
                    n_loaded = len(chunk_data)
                    raw_file_idx, raw_within_idx = raw_file_idx+1, 0
                    data, n_data = self.read_pkl(self.raw_paths[raw_file_idx])
                    chunk_data = chunk_data.concatenate(data[raw_within_idx:raw_within_idx+(batch_size-n_loaded)])
                    raw_within_idx = batch_size - n_loaded
                
                with open(chunk_file, 'wb') as f:
                    pickle.dump(chunk_data, f)
        return self.create_index_file(output_dir)
        
    def read_pkl(self, file_path, st=-1, ed=-1):
        with open(file_path, 'rb') as file:
            batch = pickle.load(file)
        Xb, Yb, length, info, As, Cs, opt, Rs, Ts = [], [], [], [], [], [], [], [], []
        cnt = 0
        if st >=0 or ed >=0:
            if ed == -1:
                batch = batch[st:]
            else:
                batch = batch[st:ed]
            print(f'batch is adjusted : {st}, {ed}', len(batch))
        for index, problem, solution in batch:  #[:500]:
            if self.phase.endswith('steinlib'):
                self.size = len(problem.nodes) 
            cnt+=1
            d = np.array(list(NodeFeature.cherrypick(problem, normalize_edge_cost=True).values()))
            
            # sort in a descending order
            indices = np.lexsort((d[:,1], d[:,0]))[::-1]
            d = d[indices]
            
            node_mapping = {i:n for i, n in enumerate(np.argsort(indices))}  # order of original index
            problem = nx.relabel_nodes(problem, node_mapping)
            solution = nx.relabel_nodes(solution, node_mapping)
            terms = []
            for t in problem.graph['Terminals']['terminals']:
                terms.append(node_mapping[t])
            problem.graph['Terminals']['terminals'] = terms
                
            Xb.append(d)
            Yb.append(self.sol_generator.run(solution, problem.graph['Terminals']['terminals']))
            l = len(Yb[-1])   # edge = (n1, n2), ..., + end token / node = [n1, n2, ...]
            length.append(l)
            info.append(problem)
            As.append(nx.to_numpy_array(problem, nodelist=range(self.size)))    # get adjacency matrix (binary)
                                                                                # 240513 sort adjacency matrix with node orders
            max_cost = max([problem.edges[n1, n2]['cost'] for n1, n2 in problem.edges])
            Cs.append(nx.to_numpy_array(problem, nodelist=range(self.size), weight='cost')/ max_cost)    # get cost map
            if self.no_label:
                if 'cost' in solution.graph['Info'].keys(): # 'test'
                    opt.append(solution.graph['Info']['cost'])
                else:   # 'valid'
                    opt.append(sum([problem.edges[e]['cost'] for e in solution.edges]))
            else:
                opt.append(sum([problem.edges[e]['cost'] for e in solution.edges]))  # sum of solution edge costs
            Rs.append(self.sol_generator.root)
            Ts.append(problem.graph['Terminals']['terminals'])

        return DataWrapper(Xb, Yb, length, info, As, Cs, opt, Rs, Ts), cnt
        
    def read_raw_paths(self, g, size, phase):
        assert g  == 'er'
        graph = 'erdos_renyi'

        if phase == 'train':
            path_dataset = f'/workspace/lab-di/squads/AIPCB/respack_v2/synthetic_datasets/traindata_0528/synthetic_STP_dataset_train_{graph}_n{size}_scipjack'
        elif phase == 'test':
            path_dataset = f'/workspace/lab-di/squads/AIPCB/respack_v2/synthetic_datasets/testdata_0528/synthetic_STP_dataset_test_{graph}_n{size}_scipjack'

        def extract_num(input_string):
            number = re.search(r'batch_(\d+)\.pkl', input_string)
            if number:
                return int(number.group(1))
            else:
                return -1
            
        path_files = glob.glob(f"{path_dataset}/*.pkl")
        path_files.sort(key=extract_num)

        return path_files

    def split_data_to_chunks(data, chunk_size, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        total_samples = len(data)
        for i in range(0, total_samples, chunk_size):
            chunk_data = data[i:i+chunk_size]
            chunk_file = os.path.join(output_dir, f'chunk_{i//chunk_size}.pkl')
            with open(chunk_file, 'wb') as f:
                pickle.dump(chunk_data, f)
    
    def create_index_file(self, chunk_dir):
        chunk_files = glob.glob(os.path.join(chunk_dir, 'chunk_*.pkl'))
        index_file = os.path.join(chunk_dir, 'index.txt')
        with open(index_file, 'w') as f:
            for chunk_file in chunk_files:
                f.write(chunk_file + '\n')
        print(f'Saving {chunk_dir} is completed !')
        return index_file


class PTR_Dataset(Dataset):
    def __init__(self, g='er', size=20, phase='train', total_data_size=100000, batch_size = 256, no_label=False):
        self.g = g
        self.size = size
        self.phase = phase
        self.total_data_size = total_data_size
        self.batch_size = batch_size
        self.no_label = no_label
        self.index_file = self.init_dataset()
        self._init_chunk()
        self.load_chunk(0)

    def init_dataset(self,):
        spd = SplitDataset(self.total_data_size, self.batch_size, g=self.g, size=self.size, phase=self.phase, no_label=self.no_label)
        output_dir=f'data_ptr/{self.phase}_{self.g}_b{self.batch_size}_n{self.size}'

        os.makedirs(output_dir, exist_ok=True)
        return spd.save(output_dir)
    
    def _init_chunk(self,):
        with open(self.index_file, 'r') as f:
            self.chunk_files = [line.strip() for line in f]
        self.chunk_size = self.batch_size
        self.current_chunk_data = []
        self.current_chunk_idx = -1
        
    def load_chunk(self, chunk_idx):
        with open(self.chunk_files[chunk_idx], 'rb') as f:
            self.current_chunk_data = pickle.load(f)
        self.current_chunk_idx = chunk_idx

    def __len__(self):
        return self.total_data_size

    def __getitem__(self, idx):
        chunk_idx = idx // self.chunk_size
        within_chunk_idx = idx % self.chunk_size

        if chunk_idx != self.current_chunk_idx:
            self.load_chunk(chunk_idx)

        return self.current_chunk_data[within_chunk_idx]

def sparse_seq_collate_fn(batch):
    batch_size = len(batch)

    try:
        sorted_seqs, sorted_labels, sorted_lengths, sorted_info, sorted_As, sorted_Cs, sorted_opts, sorted_Rs, sorted_Ts = zip(*sorted(batch, key=lambda x: x[2], reverse=True))
    except:
        try:
            sorted_seqs, sorted_labels, sorted_lengths, sorted_info, sorted_As, sorted_Cs, sorted_opts = zip(*sorted(batch, key=lambda x: x[2], reverse=True))
            sorted_Rs = None
        except:
            sorted_seqs, sorted_labels, sorted_lengths, sorted_info, sorted_As, sorted_opts = zip(*sorted(batch, key=lambda x: x[2], reverse=True))
            sorted_Cs = None
            sorted_Rs = None

    sorted_seqs = torch.FloatTensor(np.array(sorted_seqs))
    sorted_As = torch.FloatTensor(np.array(sorted_As))
    sorted_Cs = torch.FloatTensor(np.array(sorted_Cs))
    sorted_opts = np.array(sorted_opts)

    padded_seqs = [seq.resize_as_(sorted_seqs[0]) for seq in sorted_seqs]

    # (Sparse) batch_size X max_seq_len X input_dim
    seq_tensor = torch.stack(padded_seqs)

    # batch_size
    length_tensor = torch.LongTensor(sorted_lengths)
    sorted_Rs = torch.LongTensor(sorted_Rs)

    padded_labels = list(zip(*(zip_longest(*sorted_labels, fillvalue=-1))))

    # batch_size X max_seq_len (-1 padding)
    label_tensor = torch.LongTensor(padded_labels).view(batch_size, -1)   # 0-th : eos 

    return seq_tensor, label_tensor, length_tensor, sorted_info, sorted_As, sorted_Cs, sorted_opts, sorted_Rs, sorted_Ts

def convert_adj_matrix(As):
    n_instances, num_nodes = As.shape[0], As.shape[1]
    As_torch = torch.from_numpy(As)
    
    # (B, N+1, N+1)
    new_As = torch.zeros((n_instances, num_nodes+1, num_nodes+1), dtype=torch.float32)
    
    new_As[:, 1:, 1:] = As_torch

    new_As[:, 0, :] = 1
    new_As[:, :, 0] = 1
    
    return new_As
