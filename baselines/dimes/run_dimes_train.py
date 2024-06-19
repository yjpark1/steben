import copy
import datetime
import re
import os
import traceback

default_dict = {
    'device': 'cuda:0',
    'seed': 1234,
    'n_nodes': 20,
    'knn_k': 0,
    'outer_opt': 'AdamW',
    'outer_opt_lr': 5e-4,
    'outer_opt_wd': 1e-3,
    'inner_opt': 'AdamW',
    'inner_opt_lr': 5e-3,
    'inner_opt_wd': 0.0001,
    'net_units': 16,
    'net_act': 'silu',
    'emb_agg': 'mean',
    'emb_depth': 6,
    'par_depth': 3,
    'tr_batch_size': 3,
    'tr_outer_steps': 120,
    'tr_inner_steps': 15,
    'tr_inner_sample_size': 100,
    'tr_inner_greedy_size': 1,
    'graph_type': 'ER',
    'test_method': 'max',
    'cost_type': 'gaussian',
    'reward_function': 'default',
    'log_dir': 'logs/runs/dimes_{graph_type}_N{n_nodes}_T{tr_outer_steps}_{cost_type}_{test_method}@{NOW}'
}

train_schedules = [
    # {'n_nodes': 20, 'SERVER_NAME': '145', 'tr_outer_steps': 480, 'emb_depth': 6},
    # {'n_nodes': 30, 'SERVER_NAME': '146', 'tr_outer_steps': 480, 'emb_depth': 6},
    {'n_nodes': 10, 'graph_type': 'ER', 'test_method': 'softmax', 'cost_type': 'gaussian'},
    # {'n_nodes': 10, 'SERVER_NAME': '145', 'graph_type': 'ER', 'test_method': 'greedy', 'cost_type': 'uniform'},
    # {'n_nodes': 10, 'SERVER_NAME': '146', 'graph_type': 'ER', 'test_method': 'sampling', 'cost_type': 'uniform'},
    # {'n_nodes': 20, 'SERVER_NAME': '145', 'graph_type': 'ER', 'test_method': 'greedy', 'cost_type': 'gaussian'},
    # {'n_nodes': 20, 'SERVER_NAME': '146', 'graph_type': 'ER', 'test_method': 'sampling', 'cost_type': 'gaussian'},
    # {'n_nodes': 20, 'SERVER_NAME': '145', 'graph_type': 'ER', 'test_method': 'greedy', 'cost_type': 'uniform'},
    # {'n_nodes': 20, 'SERVER_NAME': '146', 'graph_type': 'ER', 'test_method': 'sampling', 'cost_type': 'uniform'},
    # {'n_nodes': 30, 'SERVER_NAME': '145', 'graph_type': 'ER', 'test_method': 'greedy', 'cost_type': 'gaussian'},
    # {'n_nodes': 30, 'SERVER_NAME': '146', 'graph_type': 'ER', 'test_method': 'sampling', 'cost_type': 'gaussian'},
    # {'n_nodes': 30, 'SERVER_NAME': '145', 'graph_type': 'ER', 'test_method': 'greedy', 'cost_type': 'uniform'},
    # {'n_nodes': 30, 'SERVER_NAME': '146', 'graph_type': 'ER', 'test_method': 'sampling', 'cost_type': 'uniform'},
    # {'n_nodes': 50, 'SERVER_NAME': '3', 'graph_type': 'Grid', 'cost_type': 'uniform'},
    # {'n_nodes': 50, 'SERVER_NAME': '4', 'graph_type': 'RR'},
    # {'n_nodes': 50, 'SERVER_NAME': '5', 'graph_type': 'WS'},
    # {'n_nodes': 50, 'SERVER_NAME': '146', 'graph_type': 'ER', 'test_method': 'sampling', 'cost_type': 'gaussian'},
    # {'n_nodes': 50, 'SERVER_NAME': '145', 'graph_type': 'ER', 'test_method': 'greedy', 'cost_type': 'uniform'},
    # {'n_nodes': 50, 'SERVER_NAME': '146', 'graph_type': 'ER', 'test_method': 'sampling', 'cost_type': 'uniform'},
    # {'n_nodes': 100, 'SERVER_NAME': '145', 'graph_type': 'ER', 'test_method': 'greedy', 'cost_type': 'gaussian'},
    # {'n_nodes': 100, 'SERVER_NAME': '146', 'graph_type': 'ER', 'test_method': 'sampling', 'cost_type': 'gaussian'},
    # {'n_nodes': 100, 'SERVER_NAME': '145', 'graph_type': 'ER', 'test_method': 'greedy', 'cost_type': 'uniform'},
    # {'n_nodes': 100, 'SERVER_NAME': '146', 'graph_type': 'ER', 'test_method': 'sampling', 'cost_type': 'uniform'},
    # {'n_nodes': 20, 'SERVER_NAME': '145', 'graph_type': 'ER'},
    # {'n_nodes': 30, 'SERVER_NAME': '145', 'graph_type': 'ER'},
    # {'n_nodes': 50, 'SERVER_NAME': '145', 'graph_type': 'ER'},
    # {'n_nodes': 100, 'SERVER_NAME': '145', 'graph_type': 'ER'},
    # {'n_nodes': 10, 'SERVER_NAME': '145', 'graph_type': 'RR'},
    # {'n_nodes': 20, 'SERVER_NAME': '145', 'graph_type': 'RR'},
    # {'n_nodes': 30, 'SERVER_NAME': '145', 'graph_type': 'RR'},
    # {'n_nodes': 50, 'SERVER_NAME': '145', 'graph_type': 'RR'},
    # {'n_nodes': 100, 'SERVER_NAME': '145', 'graph_type': 'RR'},
    # {'n_nodes': 10, 'SERVER_NAME': '146', 'graph_type': 'Grid'},
    # {'n_nodes': 20, 'SERVER_NAME': '146', 'graph_type': 'Grid'},
    # {'n_nodes': 30, 'SERVER_NAME': '146', 'graph_type': 'Grid'},
    # {'n_nodes': 50, 'SERVER_NAME': '146', 'graph_type': 'Grid'},
    # {'n_nodes': 100, 'SERVER_NAME': '146', 'graph_type': 'Grid'},
    # {'n_nodes': 10, 'SERVER_NAME': '146', 'graph_type': 'WS'},
    # {'n_nodes': 20, 'SERVER_NAME': '146', 'graph_type': 'WS'},
    # {'n_nodes': 30, 'SERVER_NAME': '146', 'graph_type': 'WS'},
    # {'n_nodes': 50, 'SERVER_NAME': '146', 'graph_type': 'WS'},
    # {'n_nodes': 100, 'SERVER_NAME': '146', 'graph_type': 'WS'},
]


def split_string_between_brackets_and_braces(s):
    # Use a regex pattern to match anything between square brackets or curly braces
    pattern = r'\{([^}]+)\}'

    # Find all matches in the input string
    matches = re.findall(pattern, s)

    return matches


def process_logdir(dict1):
    if dict1.get('log_dir', None) is None:
        return dict1
    logdir_str = dict1['log_dir']
    now1 = datetime.datetime.now()
    patterns = set(split_string_between_brackets_and_braces(logdir_str))

    for pattern in patterns:
        if pattern == 'NOW':
            logdir_str = logdir_str.replace(f'{{{pattern}}}', now1.strftime('%m%d_%H%M%S'))
        else:
            logdir_str = logdir_str.replace(f'{{{pattern}}}', f'{dict1[pattern]}')
    dict1['log_dir'] = logdir_str


if __name__ == '__main__':
    import sys
    import pathlib
    cur_dir = base_dir = pathlib.Path(__file__).parent.resolve()
    base_dir = cur_dir.parent.parent.resolve()

    sys.path.insert(0, str(base_dir))

    from baselines.dimes_tsp.inc.tsp_args import args_init
    from run_dimes import run

    server_name = os.environ.get('SERVER_NAME', None)
    for train_arg in train_schedules:
        if server_name is not None and 'SERVER_NAME' in train_arg:
            if server_name != train_arg['SERVER_NAME']:
                continue
        if 'SERVER_NAME' in train_arg:
            del train_arg['SERVER_NAME']
        default = copy.deepcopy(default_dict)
        default.update(train_arg)
        process_logdir(default)
        args = args_init(**default)
        try:
            run(args, True)
        except:
            traceback.print_exc()


