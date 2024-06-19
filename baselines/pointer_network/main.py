import torch
from model import PointerNet
from train import train
from test import test
import random
import numpy as np
import argparse
import os, json, time
from datetime import datetime

if __name__ == '__main__':
    def make_dirs(args):
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        directory_path = f"checkpoint/{timestamp}"
        os.makedirs(directory_path)
        print(timestamp, '='*100)
        args.load_id = timestamp
        with open(f'{directory_path}/config.json', 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        return args
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--tag', type=str, default='instance-synthetic_STP_dataset_1M_erdos_renyi_n20_cvxpystp')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--emb_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--n_files', type=int, default=1000) # 100이면 512 * 100 = 51200
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--no_cuda', type=bool, default=False)
    parser.add_argument('--size', type=int, default=20)
    parser.add_argument('--in_feature', type=int, default=2)
    parser.add_argument('--g', type=str, default='er')
    parser.add_argument('--load_id', type=str, default='20240513_151717')   # 만약 train==True라면, timestamp 값으로 대체됨
    parser.add_argument('--train', action='store_true', default=True)
    parser.add_argument('--save_lc', action='store_true', default=True)
    parser.add_argument('--use_index', action='store_true', default=True)
    parser.add_argument('--test_epoch', type=int, default=-1)
    parser.add_argument('--use_clip', action='store_true', default=True)
    parser.add_argument('--n_gpu', type=int, default=0)
    parser.add_argument('--use_edge_mask', action='store_true', default=False)
    parser.add_argument('--can_repeat', action='store_true', default=True)
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5)
    parser.add_argument('--memo', type=str, default='teacher forcing 0.0 / selected edge masking False / Clip True / adj sorting in node order, fix feasibility / n_files 1000 / n_epochs 500')
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device(f"cuda:{args.n_gpu}" if use_cuda else "cpu")
    
    if args.train:
        args = make_dirs(args)  
        model = PointerNet(args.in_feature, args.emb_size, args.emb_size, device, bidirectional=False, batch_first=True, use_index=args.use_index, use_edge_mask=args.use_edge_mask, can_repeat=args.can_repeat, teacher_forcing_ratio=args.teacher_forcing_ratio).to(device)
        start = time.time()
        train(model, args.n_epochs, device, args.load_id, args.g, args.size, args.batch_size, args.lr, tag=args.tag, n_files=args.n_files, save_lc=args.save_lc, use_index=args.use_index, use_clip=args.use_clip)
        print(f'[{args.load_id}] training is over. (time = {time.time() - start})')
    print(f'[{args.load_id}] test ','='*100)
    test(args.load_id, args.test_epoch, device, args.g, args.size, args.batch_size, args.lr, tag=args.tag, n_files=args.n_files)
