import torch
from model import PointerNet_TF
from train import train
from test import test
import random
import numpy as np
import argparse
import os, json, time
from datetime import datetime
from dataloader import DataWrapper  # needed for loading pickle file
from utils import get_ckpt_files

if __name__ == '__main__':
    def make_dirs(args):
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        directory_path = f"baselines/pointer_network/checkpoint/{timestamp}"
        os.makedirs(directory_path)
        print(timestamp, '='*100)
        load_id = timestamp
        with open(f'{directory_path}/config.json', 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        return args, load_id
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--tag', type=str, default='TEST')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--n_epochs', type=int, default=300)
    parser.add_argument('--emb_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--no_cuda', type=bool, default=False)
    parser.add_argument('--size', type=int, default=10)
    parser.add_argument('--g', type=str, default='er')
    parser.add_argument('--n_gpu', type=int, default=0)
    parser.add_argument('--no_label', action='store_true', default=True ,help='when solution is not included in dataset, set it as True')
    parser.add_argument('--total_data_size', type=int, default=1000000)     # 1M
    parser.add_argument('--phase', type=str, default='gaussian')
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    args, load_id = make_dirs(args)
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device(f"cuda:{args.n_gpu}" if use_cuda else "cpu")
    print(device)

    print(f'[{load_id}] train ','='*100)
    in_feature = 4 if args.phase == 'grid' else 2
    model = PointerNet_TF(in_feature, args.emb_size, args.emb_size, device).to(device)
    phase = 'train'
    start = time.time()
    train(model, args.n_epochs, device, load_id, args.g, args.size, args.batch_size, args.lr, workers=4, tag=args.tag, total_data_size=args.total_data_size, phase=phase)
    print(f'[{load_id}] train is over. (time = {time.time() - start})')

    print(f'[{load_id}] test ','='*100)
    phase = 'test'
    start = time.time()
    ckpt_file = get_ckpt_files(load_id)    # get best epoch checkpoint
    test(load_id, ckpt_file, device, args.g, args.size, args.batch_size, no_label=args.no_label, phase=phase, n_sampling=0, n_samples=10000)
    print(f'[{load_id}] test is over. (time = {time.time() - start})')