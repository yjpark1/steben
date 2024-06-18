# import networkx as nx
from dataloader import PTR_Dataset, sparse_seq_collate_fn
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from eval import Measure
import os, time
# import pickle
from tqdm import tqdm
from utils import process_data, cal_mask, binary


def train(model, n_epochs, device, load_id, g='er', size=20, batch_size=128, lr=0.001, clip_value=2.0, workers=4, tag='ptr', total_data_size=100, phase='train'):

    dataset = PTR_Dataset(g, size, phase, total_data_size, batch_size, no_label=False)
    dataset_valid = PTR_Dataset(g, size, phase.replace('train', 'valid'), 1000, batch_size, no_label=True)

    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn=sparse_seq_collate_fn)
    valid_loader = DataLoader(dataset=dataset_valid, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=sparse_seq_collate_fn)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    evaluator = Measure(load_id, 'train')
    evaluator_valid = Measure(load_id, 'valid')

    best_epoch = 0
    start = time.time()
    for epoch in tqdm(range(1, n_epochs + 1)):

        #train
        model.train()
        model.is_train = True

        for batch_idx, (x, y, target_lengths, info, As, Cs, opt, Rs, Ts) in enumerate(train_loader):
            if x.size(2) > 2:
                x = x[:,:,1:]
            b_size = x.size(0)
            max_seq_len = y.size(1)
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)
            As = As.to(device)
            Cs = Cs.to(device)
            Rs = Rs.to(device)
            max_node_size = size
            
            y, target_lengths, max_seq_len = process_data(y, target_lengths, max_seq_len)

            log_pointer_score, argmax_pointer, mask = model(x, Rs, target_lengths, max_seq_len, max_node_size, As, Cs, y)
            unrolled = log_pointer_score.view(-1, log_pointer_score.size(-1))
            loss = F.nll_loss(unrolled, y.contiguous().view(-1).long(), ignore_index=-1)
            assert not np.isnan(loss.item()), 'Model diverged with loss = NaN'
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip_value)    # weight clipping
            optimizer.step()

            mask_seq = cal_mask(b_size, max_seq_len, target_lengths)
            acc = evaluator.update_light(batch_size, y, loss.item(), argmax_pointer, mask_seq, device)

        evaluator.print_log(phase='train')
        
        # valid
        model.eval()
        model.is_train = False
        for batch_idx, (x, y, target_lengths, info, As, Cs, opt, Rs, Ts) in enumerate(valid_loader):
            target_lengths[:] = x.size(1)
            if x.size(2) > 2:
                x = x[:,:,1:]
            b_size = x.size(0)
            max_seq_len = x.size(1) ### for test
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)
            As = As.to(device)
            Cs = Cs.to(device)
            Rs = Rs.to(device)
            max_node_size = size

            y, target_lengths, max_seq_len = process_data(y, target_lengths, max_seq_len)

            log_pointer_score, argmax_pointer, mask = model(x, Rs, target_lengths, max_seq_len, max_node_size, As, Cs, y) # input_length = node_size +1, max_seq_len = max label length
            unrolled = log_pointer_score.view(-1, log_pointer_score.size(-1))
            mask_seq = cal_mask(b_size, max_seq_len, target_lengths)
            tb = binary(Ts, size)
            evaluator_valid.update(b_size, y, opt, -1, argmax_pointer, mask_seq, info, no_label=True, Rs=Rs.tolist(), Ts=Ts, update_log=True, As=As, tb=tb, device=device, n_sampling=0)

        evaluator_valid.save_epoch_log(epoch)
        _, best_epoch = min([(v, k) for k, v in evaluator_valid.epoch_log.items()])
        if best_epoch == epoch:
            print('best epoch is updated! epoch :',best_epoch)
            torch.save(model, f'baselines/pointer_network/checkpoint/{load_id}/pointer-network-BEST.pth')
            print(f'[{best_epoch}] baselines/pointer_network/checkpoint/{load_id}/pointer-network-BEST.pth is saved! (duration : {time.time() - start})')
            
