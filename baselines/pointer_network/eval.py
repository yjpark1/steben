import numpy as np
import torch
import networkx as nx
import copy

class AverageMeter(object):
	"""Computes and stores the average and current value"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


class Measure():
    def __init__(self, timestamp, test_epoch, n_sampling=0) -> None:
        self.timestamp = timestamp
        self.test_epoch = test_epoch
        self.epoch_log = {}
        self.log_book = []
        self.cur_idx = 0
        self.n_sampling=n_sampling
        
        self.loss = AverageMeter()
        self.acc = AverageMeter()
        self.fsb = AverageMeter()
        self.obj = AverageMeter()
        self.gap = AverageMeter()
        self.gap_fsb = AverageMeter()
        
    def update(self, batch_size, y, opt, loss, argmax_pointer, mask_seq, info, no_label=False, Rs=None, Ts=None, update_log=False, As=None, tb=None, device='cpu',n_sampling=0):
        self.loss.update(loss, batch_size)
        
        node_size = As.size(1)
        device = device
        edges, objs, feasibility = self.cherrypick_decoding_batch(tb, info, As, Rs, Ts, argmax_pointer, batch_size, node_size, device, post=True,n_sampling=n_sampling)
        acc = -1 if no_label else self.masked_accuracy(argmax_pointer, y, mask_seq.to(device)).item()
        self.acc.update(acc, mask_seq.int().sum().item())
        self.fsb.update(sum(feasibility) / len(feasibility), len(feasibility))
        self.obj.update(sum(objs) / len(objs), len(objs))
        self.gap.update(np.sum(np.array(objs) / opt) / len(objs), len(objs))
        np_fsb = np.array(feasibility)
        gap_fsb = (np.array(objs) / opt)[np_fsb==1]
        if len(gap_fsb) > 0:
            self.gap_fsb.update(np.sum(gap_fsb) / len(gap_fsb), len(gap_fsb))
        if update_log:
            self.log_book.append(gap_fsb)
        return acc, objs, gap_fsb, np_fsb
        
    def update_light(self, batch_size, y, loss, argmax_pointer, mask_seq, device='cpu'):
        self.loss.update(loss, batch_size)
        acc = self.masked_accuracy(argmax_pointer, y, mask_seq.to(device)).item()
        self.acc.update(acc, mask_seq.int().sum().item())
        return acc
    
    def print_log(self, phase='train'):
        if phase == 'train':
            print(f'[{self.test_epoch}]', 'Loss: {:.6f}\tAccuracy: {:.6f}'.format(self.loss.avg, self.acc.avg))
        else:
            print(f'[{self.test_epoch}]', 'Feasibility: {:.6f}\tObjective: {:.6f}\tGap: {:.6f}\tFeasible Gap: {:.6f}'.format(self.fsb.avg, self.obj.avg, self.gap.avg, self.gap_fsb.avg))
    
    def masked_accuracy(self, output, target, mask):
        """Computes a batch accuracy with a mask (for padded sequences) """
        with torch.no_grad():
            masked_output = torch.masked_select(output, mask)
            masked_target = torch.masked_select(target, mask)
            accuracy = masked_output.eq(masked_target).float().mean()
            return accuracy

    def find_edges(self, tb, Rs, info, argmax_pointer, As, batch_size, node_size, device, n_sampling=0):
        remain_terms = torch.FloatTensor(copy.copy(tb))
        cost_maps = [nx.to_numpy_array(problem, nodelist=range(node_size), weight='cost')[None,:,:] for problem in info]
        Es= torch.FloatTensor(np.concatenate(cost_maps, axis=0)).to(device)
        done = torch.ones((batch_size)).to(device)   # 
        partial_sol = torch.zeros((batch_size, node_size)).to(device)
        objs = torch.zeros(batch_size).to(device)                         # objective
        edges = []
        i=0
        if n_sampling>0:
            tb = tb.repeat(n_sampling, axis=0)
            Es = Es.unsqueeze(1).repeat(1, n_sampling,1,1).view(-1, node_size, node_size)
            remain_terms = remain_terms.unsqueeze(1).repeat(1, n_sampling,1).view(-1, node_size)
        partial_sol[range(batch_size),Rs] = 1   # add to partial sol
        remain_terms[range(batch_size),Rs] = 0   # terminals to be connected
        while min(argmax_pointer.size(1)-i, done.sum()): # batch_size, n_steps
            # index of terminals
            selected = argmax_pointer[:, i].detach().cpu().numpy()
            # connected edges
            selected_edges = As[range(batch_size), selected, :]

            # partial sol
            valid_edges = selected_edges * (partial_sol)
            
            edge_costs = Es[range(batch_size), selected, :]
            valid_edge_costs = edge_costs * valid_edges  # mask
            
            valid_edge_costs[valid_edge_costs == 0] = float('inf')  # make zero to infinite
            min_edge = valid_edge_costs.min(dim=1).values  # minimum cost edge
            
            objs += min_edge * done
            remain_terms[range(batch_size), selected] = 0   # remove remaining terminals
            partial_sol[range(batch_size), selected] = 1    # add to partial sol
            edge = np.concatenate([selected.reshape(-1,1), valid_edge_costs.argmin(axis=1).detach().cpu().numpy().reshape(-1,1)], axis=1) * (done.detach().cpu().numpy()[:,None])
            edges.append(edge)
            done[remain_terms.sum(axis=1)==0] = 0           # if feasible, then done = 0
            i+=1

        return objs.detach().cpu().numpy(), np.concatenate(edges, axis=1), done.detach().cpu().numpy()==0


    def cherrypick_decoding_batch(self, tb, info, As, Rs, Ts, argmax_pointer, batch_size, node_size, device, post=True, n_sampling=0):
        """
        extract edges from node sequence 
        """
        objs, edge_list, feasibility = self.find_edges(tb, Rs, info, argmax_pointer, As, batch_size, node_size, device, n_sampling)
        edge_output, obj_output = [], []

        for ii, (edge, obj) in enumerate(zip(edge_list, objs)):
            idx = ii if n_sampling==0 else ii // n_sampling
            g, terminals = info[idx], Ts[idx]
            pairs = edge.reshape(-1, 2)
            valid_pairs = pairs[np.any(pairs != 0, axis=1)]
            edges = [tuple(pair) for pair in valid_pairs]
            if post:    # remove redundant edges
                new_edge, did_filtering = self.filter_leaf_nodes(edges, terminals)
                if did_filtering:
                    obj = sum([g.edges[n1, n2]['cost'] for n1, n2 in new_edge])
                    edges = new_edge
            edge_output.append(edges)
            obj_output.append(obj)
        if n_sampling>0:
            np_obj = np.array(obj_output).reshape(-1, n_sampling)
            indices = np_obj.argmin(axis=1)
            edge_output = [edge_output[(i*n_sampling)+ii] for i, ii in enumerate(indices)]
            obj_output = np_obj[range(batch_size // n_sampling), indices]
            feasibility = feasibility[indices]
        assert feasibility.all(), 'all solution needs to be feasible'
        return edge_output, obj_output, feasibility

    def filter_leaf_nodes(self, edges, terminals):
        # Create a graph from the edge list
        G = nx.Graph()
        G.add_edges_from(edges)
        inessentials = []
        
        while True:
            node_list = np.array(G.nodes)
            adj_matrix = nx.to_numpy_array(G, nodelist=node_list)
            degrees = np.sum(adj_matrix, axis=1)
            leaves = node_list[np.arange(len(degrees))[degrees==1]]   #[node for node, degree in zip(sorted(G.nodes), degrees) if degree == 1]
            
            # Filter out leaves that are not terminals
            indicator = 1-np.isin(leaves, terminals).astype(int)
            not_terminal = leaves[indicator==1]
            
            if not len(not_terminal):
                break
            
            G.remove_nodes_from(not_terminal)
            inessentials.extend(not_terminal)
        
        did_filtering = len(inessentials)>0
        return list(G.edges), did_filtering

    def save_epoch_log(self, valid_epoch):
        valid_log = np.concatenate(self.log_book).mean()
        self.epoch_log[valid_epoch] = valid_log
        self.log_book = []


    def cal_feasibility(self, edges, terminals):
        nodes = dict()
        def find(x):
            if nodes[x] != x:
                return find(nodes[x])
            return x

        def union(a, b):
            a = find(a)
            b = find(b)
            if a < b:
                nodes[b] = a
            else:
                nodes[a] = b
        
        for n1, n2 in edges:
            if n1 not in nodes.keys():
                nodes[n1] = n1
            if n2 not in nodes.keys():
                nodes[n2] = n2
            union(n1, n2)
            
        roots = []
        for t in terminals:
            if t not in nodes.keys():
                nodes[t] = t
            roots.append(find(t))
        if len(list(set(roots))) == 1:
            return True
        return False

    def cal_objs(self, edges, graph_info):
        objs = 0
        for n1, n2 in edges:
            if (n1, n2) not in graph_info.edges: continue
            objs += graph_info.edges[n1, n2]['cost']
        return objs