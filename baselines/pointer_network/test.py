from dataloader import PTR_Dataset, sparse_seq_collate_fn
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from eval import Measure
from utils import process_data, cal_mask, binary


def test(load_id, ckpt_file, device, g='er', size=20, batch_size=128, workers=4, no_label=False, phase='test', n_sampling=32, n_samples=10000):
    dataset = PTR_Dataset(g, size, phase, total_data_size=n_samples, batch_size = batch_size, no_label=no_label)
    
    test_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn=sparse_seq_collate_fn)
    greedy = False if n_sampling > 0 else True
    print('greedy : ',greedy)

    epoch = 'best'
    model = torch.load(f'baselines/pointer_network/checkpoint/{load_id}/{ckpt_file}', map_location=device)
    model.device = device
    model.eval()
    model.is_train = False
    evaluator = Measure(load_id, epoch, n_sampling)
    
    for batch_idx, (x, y, target_lengths, info, As, Cs, opt, Rs, Ts) in enumerate(test_loader):
        b_size = x.size(0)
        size = x.size(1)
        max_seq_len = x.size(1) if no_label else y.size(1)  
        x = x.to(device)
        y = y.to(device)
        As = As.to(device)
        Cs = Cs.to(device)
        Rs = Rs.to(device)
        # input_lengths = input_lengths.to(device)
        max_node_size = size
        tb = binary(Ts, size)
        if not greedy:
            def expand(x,As,Cs,Rs, target_lengths):
                batch_size, node_size, feature_size = x.shape
                x_repeated = x.unsqueeze(1).repeat(1, n_sampling, 1, 1).reshape(batch_size * n_sampling, node_size, feature_size)
                As_repeated = As.unsqueeze(1).repeat(1, n_sampling, 1, 1).reshape(batch_size * n_sampling, node_size, node_size)
                Cs_repeated = Cs.unsqueeze(1).repeat(1, n_sampling, 1, 1).reshape(batch_size * n_sampling, node_size, node_size)
                Rs_repeated = Rs.unsqueeze(1).repeat(1, n_sampling).reshape(batch_size * n_sampling)
                target_repeated = target_lengths.unsqueeze(1).repeat(1, n_sampling).reshape(batch_size * n_sampling)
                return x_repeated, As_repeated, Cs_repeated, Rs_repeated, target_repeated
            x, As, Cs, Rs, target_lengths = expand(x, As, Cs, Rs, target_lengths)
            b_size *= n_sampling

        if no_label:
            target_lengths[:] = x.size(1)   # if spec['label_type'] == 'edge' else x.size(1)

        y, target_lengths, max_seq_len = process_data(y, target_lengths, max_seq_len)

        log_pointer_score, argmax_pointer, _ = model(x, Rs, target_lengths, max_seq_len, max_node_size, As, Cs, greedy=greedy) # input_length = node_size +1, max_seq_len = max label length

        unrolled = log_pointer_score.view(-1, log_pointer_score.size(-1))
        
        loss = -1 if no_label else F.nll_loss(unrolled, y.view(-1).long(), ignore_index=-1).item()
        mask_seq = cal_mask(b_size, max_seq_len, target_lengths)
        evaluator.update(b_size, y, opt, loss, argmax_pointer.clone(), mask_seq, info, no_label=no_label, Rs=Rs.tolist(), Ts=Ts, update_log=True, As=As, tb=tb, device=device,n_sampling=n_sampling)
        if batch_idx % 10 == 0:
            evaluator.print_log(phase='test')




import torch
from test import test
import random
import numpy as np
import argparse
import os, json, time
from datetime import datetime
from dataloader import DataWrapper  # needed for loading pickle file
from utils import get_ckpt_files

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--no_cuda', type=bool, default=False)
    parser.add_argument('--size', type=int, default=10)
    parser.add_argument('--g', type=str, default='er')
    parser.add_argument('--n_gpu', type=int, default=0)
    parser.add_argument('--load_id', type=str, default='ER-10')
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device(f"cuda:{args.n_gpu}" if use_cuda else "cpu")
    print(device)

    print(f'[{args.load_id}] test ','='*100)
    phase = 'test'
    start = time.time()
    ckpt_file = get_ckpt_files(args.load_id)    # get best epoch checkpoint
    test(args.load_id, ckpt_file, device, args.g, args.size, args.batch_size, no_label=True, phase=phase, n_sampling=0, n_samples=10000)
    print(f'[{args.load_id}] test is over. (time = {time.time() - start})')