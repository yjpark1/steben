import torch
from torch.autograd import Variable
import os
import numpy as np
import heapq as hq
from itertools import chain

def get_numpy_from_nonfixed_2d_array(aa, fixed_length, padding_value=0):
    rows = []
    for a in aa:
        rows.append(np.pad(list(chain(*a)) + [-1], (0, fixed_length), 'constant', constant_values=padding_value)[:fixed_length])
    return np.concatenate(rows, axis=0).reshape(-1, fixed_length) +1


def get_max_k(list, K):
    hq.heapify(list)
    cnt = 0
    output = []
    while list:
        item = hq.heappop(list)
        output.append(item)
        cnt +=1
        if cnt == K: break
    return output


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def process_data(y, target_lengths, max_seq_len):
    y = y[:,1:] # the first node is given
    target_lengths -= 1 
    max_seq_len -=1
    return y, target_lengths, max_seq_len

def cal_mask(batch_size, max_seq_len, target_lengths):
	"""
	(B, L, N+1) remove padded nodes for loss
	"""
	range_tensor = torch.arange(max_seq_len, device=target_lengths.device, dtype=target_lengths.dtype).expand(batch_size, max_seq_len).to(target_lengths.device)
	each_len_tensor = target_lengths.view(-1, 1).expand(batch_size, max_seq_len).to(target_lengths.device)
	row_mask_tensor = (range_tensor < each_len_tensor)
	mask_tensor = row_mask_tensor
	return mask_tensor

def get_ckpt_files(load_id, ckpt_dir='baselines/pointer_network/checkpoint', best=True):
    for f in os.listdir(f'{ckpt_dir}/{load_id}'):
        if not f.endswith('.pth'): continue
        if best and not f.endswith('BEST.pth'): continue
        return f

def binary(vec, n_nodes):
    output = np.zeros((len(vec), n_nodes))
    for i, v in enumerate(vec):
        output[i,v] = 1
    return output