import argparse
import datetime
from baselines.dimes_tsp.inc.utils import *
from baselines.dimes_tsp.inc.tsp_utils import *

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--device', type=str, default='cuda', help='device for torch')
    parser.add_argument('--n_nodes', type=int, default=20, help='number of nodes')
    parser.add_argument('--knn_k', default=0, type=int, help='K for KNN (-1 to turn off edge pruning)')
    parser.add_argument('--outer_opt', type=str, help='outer optimizer')
    parser.add_argument('--outer_opt_lr', type=float, help='learning rate for outer optimizer')
    parser.add_argument('--outer_opt_wd', type=float, help='weight decay for outer optimizer')
    parser.add_argument('--inner_opt', type=str, help='inner optimizer')
    parser.add_argument('--inner_opt_lr', type=float, help='learning rate for inner optimizer')
    parser.add_argument('--inner_opt_wd', type=float, help='weight decay for inner optimizer')
    parser.add_argument('--net_units', type=int, help='number of units in the neural network')
    parser.add_argument('--net_act', type=str, help='activation function in the neural network')
    parser.add_argument('--emb_agg', type=str, choices=['add', 'max', 'mean'], help='aggregation function of EmbNet')
    parser.add_argument('--emb_depth', type=int, help='number of layers in EmbNet')
    parser.add_argument('--par_depth', type=int, help='number of layers in ParNet')
    parser.add_argument('--tr_gamma', type=float, default=0.99, help='number of layers in ParNet')
    parser.add_argument('--tr_batch_size', type=int, help='batch size for training')
    parser.add_argument('--tr_outer_steps', type=int, help='outer steps for training')
    parser.add_argument('--tr_inner_steps', type=int, help='inner steps for training')
    parser.add_argument('--tr_inner_sample_size', type=int, help='inner sample size for training')
    parser.add_argument('--tr_inner_greedy_size', type=int, help='inner greedy size for training')
    parser.add_argument('--te_net', type=int, help='which net to test')
    parser.add_argument('--te_range_l', type=int, help='l of test graphs id range [l, r)')
    parser.add_argument('--te_range_r', type=int, help='r of test graphs id range [l, r)')
    parser.add_argument('--te_batch_size', type=int, help='batch size for computing emb for testing')
    parser.add_argument('--te_tune_steps', type=int, help='tuning steps in testing')
    parser.add_argument('--te_tune_sample_size', type=int, help='sample size for tuning in testing')
    parser.add_argument('--te_sample_size', type=int, help='sample size for finding solutions in testing')
    parser.add_argument('--te_sample_tau', type=float, help='temperature parameter for softmax sampling in testing')
    parser.add_argument('--save_name', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--graph_emb_dim', type=int, default=8, help='STP graph to embedding vector dim')
    parser.add_argument('--filename_for_dqn_weights', type=str, default='dimes_weight')
    parser.add_argument('--cost_type', default='gaussian', choices=['gaussian', 'uniform'])
    parser.add_argument('--graph_type', default='ER', choices=['ER', 'Grid', 'RR', 'WS'])
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--pruning', type=int, default=1)
    parser.add_argument('--test_terminal', type=int, default=1)
    parser.add_argument('--test_method', default='max', choices=['max', 'softmax'])
    parser.add_argument('--temperature', default=1.0, type=float)
    parser.add_argument('--reward_function', default='default', choices=['default'])
    return parser

def args_prep(args):
    random.seed(args.seed)
    np.random.seed(args.seed + 1)
    torch.manual_seed(args.seed + 2)
    args.cuda = False
    if torch.cuda.is_available():
        args.cuda = True
        torch.cuda.manual_seed(args.seed + 3)

    args.device = torch.device(args.device)

    if 'n_nodes_min' not in args:
        args.n_nodes_min = args.n_nodes
    if 'n_nodes_max' not in args:
        args.n_nodes_max = args.n_nodes

    if args.knn_k <= 0:
        args.knn_k = None

    args.outer_opt_fn = lambda pars: getattr(optim, args.outer_opt)(pars, lr = args.outer_opt_lr, weight_decay = args.outer_opt_wd)
    args.inner_opt_fn = lambda pars: getattr(optim, args.inner_opt)(pars, lr = args.inner_opt_lr, weight_decay = args.inner_opt_wd)

    args.net_act_fn = getattr(F, args.net_act)
    args.emb_agg_fn = getattr(gnn, f'global_{args.emb_agg}_pool')
    args.graph_emb_dim = 2
    args.save_name = tsp_save_name(args, args.save_name)

    os.makedirs(args.output_dir, exist_ok = True)
    # args.log_dir = f"logs/runs/"
    args.torch_deterministic = True
    if getattr(args, 'log_dir', None) is None:
        args.log_dir = f'logs/runs/dimes@{datetime.datetime.now().strftime("%m%d_%H%M%S")}/'

    return args

def args_to_list(args):
    return [[f'--{k}', str(v)] for k, v in args.items()]

def args_init(**kwargs):
    return args_prep(Dict(vars(args_parser().parse_args(sum(args_to_list(kwargs), []) if len(kwargs) else None))))
